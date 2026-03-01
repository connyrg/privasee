import requests
import base64

def get_databricks_secret(workspace_url:str, 
                          scope_name: str, 
                          key: str, 
                          databricks_token: str) -> str:

    url = f"{workspace_url}/api/2.0/secrets/get"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {databricks_token}",
    }
    params = {
        'scope': scope_name,
        'key': key,
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    decoded_bytes = base64.b64decode(response.json()['value'])
    value = decoded_bytes.decode('utf-8')
    return value


def generate_sp_token(sp_workspace_url, sp_client_id, sp_client_secret):
    # Retrieve Service Principal OAuth token
    token_endpoint = f"{sp_workspace_url}/oidc/v1/token"
    data = {
        "grant_type": "client_credentials",
        "scope": "all-apis"
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response = requests.post(url=token_endpoint, headers=headers, data=data, auth=(sp_client_id, sp_client_secret))
    response.raise_for_status()

    token = response.json()["access_token"]

    return token