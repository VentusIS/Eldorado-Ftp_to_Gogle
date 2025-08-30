#!/usr/bin/env python3
import os, io, sys, json, logging, re
from typing import Optional, List, Dict, Tuple
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

import ftplib
try:
    import paramiko
except ImportError:
    paramiko = None

import pandas as pd
from lxml import etree

import gspread
from google.oauth2.service_account import Credentials

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

TARGET_ORDER = ["Order ID", "Amount", "Date", "Tracking Number", "Shipping Method"]

# ---------------- Google helpers ----------------
def _load_service_account_credentials():
    sa_json = os.getenv("GOOGLE_SA_JSON")
    sa_file = os.getenv("GOOGLE_SA_JSON_FILE")
    if sa_file:
        if not os.path.exists(sa_file):
            raise RuntimeError(f"GOOGLE_SA_JSON_FILE path does not exist: {sa_file}")
        with open(sa_file, "r") as f:
            info = json.load(f)
    elif sa_json:
        try:
            info = json.loads(sa_json)
        except json.JSONDecodeError:
            raise RuntimeError("GOOGLE_SA_JSON is not valid JSON. Use GOOGLE_SA_JSON_FILE instead.")
    else:
        raise RuntimeError("Provide either GOOGLE_SA_JSON (inline) or GOOGLE_SA_JSON_FILE (path).")
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    return Credentials.from_service_account_info(info, scopes=scopes)

def _open_sheet(spreadsheet_id: str, worksheet_name: str):
    creds = _load_service_account_credentials()
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(spreadsheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=1, cols=len(TARGET_ORDER))
        ws.update([TARGET_ORDER])
    return sh, ws

def _get_existing_order_ids(ws) -> set:
    values = ws.get_all_values()
    if not values:
        ws.update([TARGET_ORDER])
        return set()
    return {row[0].strip() for row in values[1:] if row and len(row) >= 1 and row[0].strip()}

def _append_df(ws, df: pd.DataFrame):
    if df is None or df.empty:
        logger.info("Nothing to append.")
        return
    data = df.astype(object).where(pd.notnull(df), "").values.tolist()
    CHUNK = 500
    for i in range(0, len(data), CHUNK):
        ws.append_rows(data[i:i+CHUNK], value_input_option="RAW")
    logger.info("Appended %d new rows.", len(data))

# ---------------- FTP / SFTP ----------------
def _ftp_connect(host: str, port: int, user: str, password: str) -> ftplib.FTP:
    ftp = ftplib.FTP()
    ftp.connect(host, port, timeout=60)
    ftp.login(user, password)
    return ftp

def _ftp_list_files(ftp: ftplib.FTP, remote_dir: str) -> List[str]:
    ftp.cwd(remote_dir)
    try:
        entries = [name for name, facts in ftp.mlsd() if facts.get("type") == "file"]
    except Exception:
        entries = ftp.nlst()
    return [e for e in entries if not e.startswith(".")]

def _ftp_fetch(ftp: ftplib.FTP, filename: str) -> bytes:
    bio = io.BytesIO()
    ftp.retrbinary(f"RETR {filename}", bio.write)
    return bio.getvalue()

def _sftp_connect(host: str, port: int, user: str, password: str):
    if paramiko is None:
        raise RuntimeError("paramiko is required for SFTP. Add it to requirements.txt and set PROTOCOL=sftp.")
    transport = paramiko.Transport((host, port))
    transport.connect(username=user, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    return sftp, transport

def _sftp_list_files(sftp, remote_dir: str) -> List[str]:
    sftp.chdir(remote_dir)
    files = [f for f in sftp.listdir_attr(".") if not str(f.filename).startswith(".")]
    files.sort(key=lambda x: x.st_mtime)
    return [f.filename for f in files]

def _sftp_fetch(sftp, filename: str) -> bytes:
    with sftp.open(filename, "rb") as f:
        return f.read()

# ---------------- XML helpers ----------------
NOISE_TAGS = {
    "response","response_code","responsecode","response_message","responsemessage",
    "status","message","meta","metadata","errors","error","count","success","code","result","results","info"
}
LIKELY_CONTAINERS = {"Orders","orders","OrderList","orderList","Shipments","shipments","Root","root","Data","data"}
COMMON_RECORD_TAGS = {"Order","ORDER","order","Record","RECORD","record","Row","ROW","row","Item","ITEM","item"}
FIELD_LIKE_TOKENS = {"web_order_number","order","id","date","track","awb","consign","amount","total","price","shipping","carrier","service","method","tracking"}

def _flatten_elem(elem: etree._Element, prefix: str = "", out: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Recursively flatten an element:
    - Text at any depth -> key like 'parent.child'
    - Attributes -> key like 'tag@attr'
    """
    if out is None:
        out = {}
    if not isinstance(elem.tag, str):
        return out

    name = elem.tag
    key_prefix = f"{prefix}.{name}" if prefix else name

    # element text
    text = (elem.text or "").strip()
    if text:
        out[key_prefix] = text

    # attributes
    for attr, val in (elem.attrib or {}).items():
        if val:
            out[f"{key_prefix}@{attr}"] = str(val).strip()

    # recurse
    for ch in elem:
        _flatten_elem(ch, key_prefix, out)

    return out

def _score_parent_for_order_fields(elem: etree._Element) -> int:
    score = 0
    for child in elem:
        if not isinstance(child.tag, str):
            continue
        t = child.tag.lower()
        if any(tok in t for tok in FIELD_LIKE_TOKENS):
            score += 1
    return score

def _detect_container(root: etree._Element) -> Optional[etree._Element]:
    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
        if elem.tag in LIKELY_CONTAINERS and len(list(elem)) > 0:
            return elem
    return None

def _detect_record_tag(root: etree._Element, hint: Optional[str]) -> Tuple[Optional[str], Optional[etree._Element]]:
    if hint:
        matches = root.findall(f".//{hint}")
        if matches:
            return hint, None

    container = _detect_container(root)
    if container is not None:
        child_counter = Counter()
        for ch in container:
            if isinstance(ch.tag, str) and ch.tag.lower() not in NOISE_TAGS:
                child_counter[ch.tag] += 1

        if child_counter:
            for name in COMMON_RECORD_TAGS:
                if child_counter.get(name, 0) > 1:
                    return name, None
            # if looks field-like → one order per file
            fieldish = sum(1 for t in child_counter if any(tok in t.lower() for tok in FIELD_LIKE_TOKENS))
            if fieldish >= 1:
                return None, container
            most_tag, cnt = child_counter.most_common(1)[0]
            if cnt > 1:
                return most_tag, None
            else:
                return None, container

    # fallbacks
    tag_counts = Counter()
    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
        tag = elem.tag
        if tag.lower() in NOISE_TAGS:
            continue
        tag_counts[tag] += 1
    for pref in COMMON_RECORD_TAGS:
        if tag_counts.get(pref, 0) > 1:
            return pref, None

    best = None
    best_score = 0
    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
        if elem.tag.lower() in NOISE_TAGS:
            continue
        if len(list(elem)) == 0:
            continue
        s = _score_parent_for_order_fields(elem)
        if s > best_score:
            best_score = s
            best = elem
    if best is not None:
        return None, best

    return None, None

def _map_row(flat: Dict[str, str]) -> Dict[str, str]:
    # Normalize keys to lower for searching
    items = {k.lower(): v for k, v in flat.items()}

    def get_first(exacts: List[str], contains: List[str]) -> Optional[str]:
        for k in exacts:
            if k in items and items[k]:
                return items[k]
        for needle in contains:
            for k, v in items.items():
                if needle in k and v:
                    return v
        return None

    # Order ID
    order_id = get_first(
        ["web_order_number","order_id","orderid","order_number","order-no","id"],
        ["web_order","order","ref","number"]
    )

    # Amount (much broader)
    amount = get_first(
        ["amount","order_amount","orderamount","total","order_total","ordertotal","grand_total","grandtotal",
         "subtotal","price","unit_price","unitprice","amount@value","total@value","price@value"],
        ["amount","total","price","sum","subtotal","amt"]
    )

    # Date
    date = get_first(
        ["date","order_date","orderdate","created","create_date","datetime","timestamp"],
        ["date","created","time"]
    )

    # Tracking
    tracking = get_first(
        ["tracking_number","tracking","awb","consignment","consign","tn"],
        ["track","awb","consign","tracking"]
    )

    # Method
    method = get_first(
        ["shipping_method","ship_method","shipping","service","carrier","method","ship_via","shipvia"],
        ["shipping","service","carrier","method"]
    )

    # Heuristics for amount text patterns
    def find_money_candidate() -> Optional[str]:
        for k, v in items.items():
            if not v:
                continue
            s = v.strip().replace(",", "")
            # allow $ or leading currency codes
            s = s.replace("$", "")
            # match 1-4 digits with optional .2 decimals (5, 5.75, 1234, 29.22)
            if re.fullmatch(r"[A-Za-z]{0,3}\s*\d{1,4}(\.\d{1,2})?", v.strip()) or re.fullmatch(r"\d{1,4}(\.\d{1,2})?", s):
                # skip clear non-amounts (tracking: 12+ digits)
                digits = "".join(ch for ch in s if ch.isdigit())
                if 1 <= len(digits) <= 8:
                    return v.strip()
        return None

    # Swap fixes Amount ↔ Method
    def is_money_like(x: str) -> bool:
        if not x:
            return False
        s = x.strip().replace(",", "").replace("$", "")
        return re.fullmatch(r"\d{1,4}(\.\d{1,2})?", s) is not None

    carriers = ("usps","ups","fedex","dhl","ground","express","priority","advantage","smartpost","home delivery")

    if (not amount or not is_money_like(amount)) and method and is_money_like(method):
        amount, method = method, ""

    if amount and not is_money_like(amount) and not method and any(w in amount.lower() for w in carriers):
        method, amount = amount, ""

    if not amount:
        amount = find_money_candidate()

    if not amount:
        logger.debug("Amount still missing; record keys: %s", ", ".join(sorted(items.keys())))

    return {
        "Order ID":        (order_id or "").strip(),
        "Amount":          (amount or "").strip(),
        "Date":            (date or "").strip(),
        "Tracking Number": (tracking or "").strip(),
        "Shipping Method": (method or "").strip(),
    }

def _xml_records_to_df(xml_bytes: bytes, record_tag_hint: Optional[str]) -> pd.DataFrame:
    try:
        root = etree.fromstring(xml_bytes)
    except Exception as e:
        raise RuntimeError(f"Invalid XML: {e}")

    record_tag, container_elem = _detect_record_tag(root, record_tag_hint)

    rows: List[Dict[str, str]] = []
    if container_elem is not None and record_tag is None:
        flat = _flatten_elem(container_elem)
        mapped = _map_row(flat)
        rows.append(mapped)
    else:
        elems = root.findall(f".//{record_tag}") if record_tag else []
        for rec in elems:
            flat = _flatten_elem(rec)
            mapped = _map_row(flat)
            rows.append(mapped)

    df = pd.DataFrame(rows, dtype=str)
    if not df.empty:
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        df = df[df["Order ID"] != ""]
        df = df.drop_duplicates(subset=["Order ID"])
    return df

def _normalize_orders_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=TARGET_ORDER)
    out = df.reindex(columns=TARGET_ORDER, fill_value="")
    out = out.drop_duplicates(subset=["Order ID"])
    return out

# ---------------- Main ----------------
def main():
    protocol = os.getenv("PROTOCOL", "ftp").lower()
    host = os.getenv("FTP_HOST")
    port = int(os.getenv("FTP_PORT") or (22 if protocol == "sftp" else 21))
    user = os.getenv("FTP_USER")
    password = os.getenv("FTP_PASSWORD")
    remote_dir = os.getenv("FTP_REMOTE_DIR", "/")
    explicit_filename = os.getenv("FTP_FILENAME")

    spreadsheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")
    worksheet_name = os.getenv("GOOGLE_SHEETS_WORKSHEET", "Sheet1")
    record_tag_hint = (os.getenv("XML_RECORD_TAG") or "").strip() or None

    if not all([host, user, password, spreadsheet_id]):
        missing = [k for k, v in [
            ("FTP_HOST", host), ("FTP_USER", user), ("FTP_PASSWORD", password),
            ("GOOGLE_SHEETS_SPREADSHEET_ID", spreadsheet_id),
        ] if not v]
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    logger.info("Starting job: protocol=%s, host=%s, remote_dir=%s", protocol, host, remote_dir)

    if protocol == "ftp":
        ftp = _ftp_connect(host, port, user, password)
        try:
            file_names = [explicit_filename] if explicit_filename else _ftp_list_files(ftp, remote_dir)
        finally:
            try: ftp.quit()
            except Exception: pass
    else:
        sftp, transport = _sftp_connect(host, port, user, password)
        try:
            file_names = [explicit_filename] if explicit_filename else _sftp_list_files(sftp, remote_dir)
        finally:
            try: sftp.close()
            except Exception: pass
            try: transport.close()
            except Exception: pass

    xml_files = [fn for fn in file_names if fn and fn.lower().endswith(".xml")]
    logger.info("Found %d files total; using %d XML files.", len(file_names), len(xml_files))
    if not xml_files:
        raise RuntimeError("No .xml files found in the remote directory.")

    batches: List[pd.DataFrame] = []
    sftp, transport = _sftp_connect(host, port, user, password) if protocol == "sftp" else (None, None)
    ftp = _ftp_connect(host, port, user, password) if protocol == "ftp" else None
    try:
        if protocol == "sftp":
            sftp.chdir(remote_dir)
            for fn in xml_files:
                try:
                    logger.info("Downloading: %s", fn)
                    content = _sftp_fetch(sftp, fn)
                    df_raw = _xml_records_to_df(content, record_tag_hint)
                    df_norm = _normalize_orders_df(df_raw)
                    if not df_norm.empty:
                        batches.append(df_norm)
                except Exception as e:
                    preview = content[:150].decode(errors="ignore") if 'content' in locals() else ""
                    logger.warning("Skipping '%s' due to error: %s | preview: %s", fn, e, preview.replace("\n"," ")[:140])
        else:
            ftp.cwd(remote_dir)
            for fn in xml_files:
                try:
                    logger.info("Downloading: %s", fn)
                    content = _ftp_fetch(ftp, fn)
                    df_raw = _xml_records_to_df(content, record_tag_hint)
                    df_norm = _normalize_orders_df(df_raw)
                    if not df_norm.empty:
                        batches.append(df_norm)
                except Exception as e:
                    preview = content[:150].decode(errors="ignore") if 'content' in locals() else ""
                    logger.warning("Skipping '%s' due to error: %s | preview: %s", fn, e, preview.replace("\n"," ")[:140])
    finally:
        if ftp:
            try: ftp.quit()
            except Exception: pass
        if sftp:
            try: sftp.close()
            except Exception: pass
        if transport:
            try: transport.close()
            except Exception: pass

    df_all = pd.concat(batches, ignore_index=True).drop_duplicates(subset=["Order ID"]) if batches else pd.DataFrame(columns=TARGET_ORDER)
    logger.info("Parsed %d orders from XML files.", len(df_all))

    _, ws = _open_sheet(spreadsheet_id, worksheet_name)
    existing_ids = _get_existing_order_ids(ws)
    new_df = df_all[~df_all["Order ID"].isin(existing_ids)]
    if new_df.empty:
        logger.info("No new orders to append. All Order IDs already present.")
    else:
        _append_df(ws, new_df)
    logger.info("Done. %d new orders appended.", 0 if new_df.empty else len(new_df))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Job failed: %s", e)
        sys.exit(1)
