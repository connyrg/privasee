"""Azure Document Intelligence (ADI) Utility Functions

Stateless utility functions for interacting with Azure Document Intelligence API
through APIM gateway using direct HTTP requests.
"""

import requests
import base64
import time
from typing import Dict, Optional, Any


def generate_adi_token(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    api_app_id_uri: str = 'api://aeddc053-d47f-4352-9977-4313e0625905',
    proxies: Optional[Dict[str, str]] = {}
) -> str:
    """Generate OAuth token for Azure Document Intelligence API.
    
    Args:
        tenant_id: Azure tenant ID
        client_id: Service principal client ID
        client_secret: Service principal client secret
        api_app_id_uri: API application ID URI (default: ADI app URI)
        proxies: Proxy configuration dict with 'http' and 'https' keys
        
    Returns:
        OAuth access token string
        
    Raises:
        requests.HTTPError: If token generation fails
    """
    token_endpoint = f"https://login.microsoftonline.com/{tenant_id}/oauth2/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "client_credentials",
        "resource": api_app_id_uri
    }
    
    response = requests.post(
        url=token_endpoint,
        headers=headers,
        data=data,
        auth=(client_id, client_secret),
        # proxies=proxies
    )
    response.raise_for_status()

    return response.json()["access_token"]


def encode_file_to_base64(file_path: str) -> str:
    """Encode a file to base64 string.
    
    Args:
        file_path: Path to the file to encode
        
    Returns:
        Base64 encoded string
    """
    with open(file_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode('utf-8')
    return encoded_string


def analyze_document(
    file_path: str,
    token: str,
    endpoint_url: str,
    appspace_id: str,
    model_id: str = "prebuilt-layout",
    pages: str = "1",
    locale: str = "en-US",
    output_content_format: str = "markdown",
    api_version: str = "2024-11-30",
    proxies: Optional[Dict[str, str]] = None
) -> str:
    """Submit document analysis request to ADI API.
    
    Args:
        file_path: Path to the document file to analyze
        token: OAuth bearer token
        endpoint_url: ADI API endpoint (e.g., 'https://apim-nonprod-idp.azure-api.net/documentintelligence/documentModels/{model}:analyze')
        appspace_id: AppspaceId header value for APIM
        model_id: ADI model to use (default: "prebuilt-layout")
        pages: Pages to analyze (default: "1")
        locale: Document locale (default: "en-US")
        output_content_format: Output format (default: "markdown")
        api_version: API version (default: "2024-11-30")
        proxies: Proxy configuration dict with 'http' and 'https' keys
        
    Returns:
        Operation-Location URL for polling results
        
    Raises:
        requests.HTTPError: If API request fails
        ValueError: If Operation-Location header is missing
    """
    # Encode document to base64
    base64_string = encode_file_to_base64(file_path)
    
    # Submit analysis request
    url = endpoint_url.format(model=model_id)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "AppspaceId": appspace_id
    }
    params = {
        "_overload": "analyzeDocument",
        "api-version": api_version,
        "pages": pages,
        "outputContentFormat": output_content_format,
        "locale": locale,
        "features": None
    }
    body = {
        "base64Source": base64_string
    }

    response = requests.post(
        url=url, 
        headers=headers, 
        params=params, 
        json=body
    )
    response.raise_for_status()
    
    # Extract result location from response headers
    result_location = response.headers.get('Operation-Location')
    if not result_location:
        raise ValueError("No Operation-Location header in response")
    
    return result_location


def get_analysis_result(
    result_location: str,
    token: str,
    appspace_id: str,
    poll_interval: int = 2,
    max_retries: int = 60,
    proxies: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Poll for analysis results until completion.
    
    Args:
        result_location: Operation-Location URL from analyze_document()
        token: OAuth bearer token
        appspace_id: AppspaceId header value for APIM
        poll_interval: Seconds between polling attempts (default: 2)
        max_retries: Maximum polling attempts (default: 60)
        proxies: Proxy configuration dict with 'http' and 'https' keys
        
    Returns:
        Analysis result dictionary with 'status' and 'analyzeResult' keys
        
    Raises:
        requests.HTTPError: If API request fails
        TimeoutError: If polling exceeds max_retries
        RuntimeError: If analysis fails
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "AppspaceId": appspace_id
    }
    
    for attempt in range(max_retries):
        time.sleep(poll_interval)
        
        response = requests.get(
            url=result_location, 
            headers=headers
        )
        response.raise_for_status()
        result_json = response.json()
        
        status = result_json.get('status')
        if status == 'succeeded':
            return result_json
        elif status == 'failed':
            raise RuntimeError(f"Document analysis failed: {result_json}")
        # else: status is 'running' or 'notStarted', continue polling
    
    raise TimeoutError(f"Document analysis timed out after {max_retries * poll_interval} seconds")


def analyze_document_complete(
    file_path: str,
    token: str,
    endpoint_url: str,
    appspace_id: str,
    model_id: str = "prebuilt-layout",
    pages: str = "1",
    locale: str = "en-US",
    output_content_format: str = "markdown",
    api_version: str = "2024-11-30",
    poll_interval: int = 2,
    max_retries: int = 60,
    proxies: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Analyze a document and wait for results (combines analyze + polling).
    
    Args:
        file_path: Path to the document file to analyze
        token: OAuth bearer token
        endpoint_url: ADI API endpoint
        appspace_id: AppspaceId header value for APIM
        model_id: ADI model to use (default: "prebuilt-layout")
        pages: Pages to analyze (default: "1")
        locale: Document locale (default: "en-US")
        output_content_format: Output format (default: "markdown")
        api_version: API version (default: "2024-11-30")
        poll_interval: Seconds between polling attempts (default: 2)
        max_retries: Maximum polling attempts (default: 60)
        proxies: Proxy configuration dict with 'http' and 'https' keys
        
    Returns:
        Analysis result dictionary with 'status' and 'analyzeResult' keys
        
    Raises:
        requests.HTTPError: If API request fails
        TimeoutError: If polling exceeds max_retries
        RuntimeError: If analysis fails
    """
    # Submit analysis request
    result_location = analyze_document(
        file_path=file_path,
        token=token,
        endpoint_url=endpoint_url,
        appspace_id=appspace_id,
        model_id=model_id,
        pages=pages,
        locale=locale,
        output_content_format=output_content_format,
        api_version=api_version,
        proxies=proxies
    )
    
    # Poll for results
    result = get_analysis_result(
        result_location=result_location,
        token=token,
        appspace_id=appspace_id,
        poll_interval=poll_interval,
        max_retries=max_retries,
        proxies=proxies
    )
    
    return result
