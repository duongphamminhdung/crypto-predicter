#!/usr/bin/env python3
"""
Update predict_live.py thresholds in Google Drive from local changes
"""

import os
import json
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io
import difflib

# Configuration - Update these
SERVICE_ACCOUNT_FILE = '../service-account.json'
DRIVE_FILE_ID = 'YOUR_PREDICT_LIVE_FILE_ID'  # Get from Drive URL

def get_drive_service():
    """Initialize Google Drive API service"""
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=['https://www.googleapis.com/auth/drive.file']
    )
    return build('drive', 'v3', credentials=credentials)

def download_file(service, file_id, filename):
    """Download file from Drive"""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Downloading {filename}: {int(status.progress() * 100)}%")

    fh.seek(0)
    content = fh.read().decode('utf-8')

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Downloaded: {filename}")
    return content

def upload_file(service, file_id, filename):
    """Upload file to Drive"""
    media = MediaFileUpload(filename, mimetype='text/plain')

    file_metadata = {'name': filename}

    service.files().update(
        fileId=file_id,
        media_body=media
    ).execute()

    print(f"Uploaded: {filename}")

def show_diff(local_file, drive_content):
    """Show differences between local and Drive versions"""
    with open(local_file, 'r', encoding='utf-8') as f:
        local_content = f.read()

    local_lines = local_content.splitlines()
    drive_lines = drive_content.splitlines()

    diff = list(difflib.unified_diff(
        drive_lines,
        local_lines,
        fromfile='Drive version',
        tofile='Local version',
        lineterm=''
    ))

    if diff:
        print("\n" + "="*60)
        print("CHANGES TO UPLOAD:")
        print("="*60)
        for line in diff:
            if line.startswith('+'):
                print(f"\033[92m{line}\033[0m")  # Green for additions
            elif line.startswith('-'):
                print(f"\033[91m{line}\033[0m")  # Red for deletions
            else:
                print(line)
    else:
        print("No differences found!")

    return len(diff) > 0

def main():
    """Main update function"""
    print("Updating predict_live.py thresholds in Google Drive...")

    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        print(f"Error: Service account file not found: {SERVICE_ACCOUNT_FILE}")
        print("Please download your service-account.json from Google Cloud Console")
        return

    local_file = 'predict_live.py'
    if not os.path.exists(local_file):
        print(f"Error: Local file not found: {local_file}")
        return

    try:
        service = get_drive_service()

        # Download current Drive version
        print("Downloading current version from Drive...")
        drive_content = download_file(service, DRIVE_FILE_ID, f'{local_file}.drive')

        # Show differences
        has_changes = show_diff(local_file, drive_content)

        if has_changes:
            response = input("\nUpload local changes to Drive? (y/N): ").lower().strip()
            if response == 'y':
                upload_file(service, DRIVE_FILE_ID, local_file)
                print("✅ Successfully updated predict_live.py in Google Drive!")
                print("🔄 Colab will use the new thresholds on next restart.")
            else:
                print("Upload cancelled.")
        else:
            print("✅ Local and Drive versions are identical.")

    except Exception as e:
        print(f"Error during update: {e}")

if __name__ == '__main__':
    main()