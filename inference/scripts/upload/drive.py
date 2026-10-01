"""Google Shared Drive client.

Two rules the API enforces and this module never omits:
  - every call passes supportsAllDrives=True
  - every listing also passes includeItemsFromAllDrives=True, corpora='drive'
    and the driveId
A Shared Drive is invisible without them, and the failure mode is an empty
result rather than an error.

The service account is a Content manager: canEdit and canTrash are True,
canDelete is False. Overwrite is therefore find-by-name then files.update,
never delete-then-create - which is the idempotent form R15 wants anyway.
"""

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))   # scripts/ for pins, common

import mimetypes
from pathlib import Path

FOLDER_MIME = "application/vnd.google-apps.folder"


def build_service():
    import os
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from common import load_env
    load_env()
    creds = service_account.Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
        scopes=["https://www.googleapis.com/auth/drive"])
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _q(parent_id, name=None, folders_only=False):
    q = f"'{parent_id}' in parents and trashed=false"
    if name is not None:
        q += f" and name='{name}'"
    if folders_only:
        q += f" and mimeType='{FOLDER_MIME}'"
    return q


def list_folder(svc, drive_id, folder_id):
    return svc.files().list(
        corpora="drive", driveId=drive_id, includeItemsFromAllDrives=True,
        supportsAllDrives=True, q=_q(folder_id),
        fields="files(id,name,size,mimeType)", pageSize=1000).execute().get("files", [])


def find_file(svc, drive_id, parent_id, name):
    hits = svc.files().list(
        corpora="drive", driveId=drive_id, includeItemsFromAllDrives=True,
        supportsAllDrives=True, q=_q(parent_id, name),
        fields="files(id,name,size,mimeType)", pageSize=10).execute().get("files", [])
    return hits[0] if hits else None


def ensure_folder(svc, drive_id, parent_id, name):
    """Folder id for `name` under `parent_id`, creating it only if absent."""
    hit = find_file(svc, drive_id, parent_id, name)
    if hit:
        return hit["id"]
    created = svc.files().create(
        body={"name": name, "parents": [parent_id], "mimeType": FOLDER_MIME},
        fields="id,name", supportsAllDrives=True).execute()
    return created["id"]


def ensure_path(svc, drive_id, root_id, parts):
    fid = root_id
    for part in parts:
        fid = ensure_folder(svc, drive_id, fid, part)
    return fid


def put_file(svc, drive_id, parent_id, path):
    """Upload one file, updating in place when the name already exists."""
    from googleapiclient.http import MediaFileUpload
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    media = MediaFileUpload(str(path), mimetype=mime, resumable=True)
    hit = find_file(svc, drive_id, parent_id, path.name)
    if hit:
        return svc.files().update(fileId=hit["id"], media_body=media,
                                  fields="id,name", supportsAllDrives=True).execute()
    return svc.files().create(body={"name": path.name, "parents": [parent_id]},
                              media_body=media, fields="id,name",
                              supportsAllDrives=True).execute()
