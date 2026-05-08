import gspread
import streamlit as st
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


def _get_credentials() -> Credentials:
    if "GOOGLE_CREDENTIALS" in st.secrets:
        creds_dict = dict(st.secrets["GOOGLE_CREDENTIALS"])
        return Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return Credentials.from_service_account_file("credentials.json", scopes=SCOPES)


def load_sheet_data(sheet_id: str) -> list[dict]:
    creds = _get_credentials()
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.worksheet("VendorDB")
    records = worksheet.get_all_records()
    print(f"Loaded {len(records)} records from VendorDB")
    return records


def records_to_chunks(records: list[dict]) -> list[str]:
    chunks = []
    for i, record in enumerate(records):
        lines = [f"Vendor #{i + 1}:"]
        for key, value in record.items():
            if value != "" and value is not None:
                lines.append(f"  {key}: {value}")
        chunks.append("\n".join(lines))
    return chunks
