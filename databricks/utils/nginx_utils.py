import asyncio
import httpx
import json
import time

import os
import requests
import json
from typing import Dict

import warnings
import logging

# Get a logger instance using the standard __name__ variable
logger = logging.getLogger(__name__)

from .databricks_utils import generate_sp_token 


def authenticate_and_get_headers(
    openai_key, proxy_client_id, proxy_client_secret, workspace_url
) -> Dict[str, str]:
    token = generate_sp_token(workspace_url, proxy_client_id, proxy_client_secret)

    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "api-key": openai_key,
    }

def http_client_factory(workspace_url, openai_apikey=None, proxy_client_id=None, proxy_client_secret=None, api_version=None):
    """
    Create configured `httpx` clients (sync and async) that automatically prepare
    requests for nginx reverse proxy expecting authenticated headers to be used with 
    vanilla AzureChatOpenAI.

    This factory wires a request hook that:
      - Injects authentication and proxy headers required by the nginx reverse proxy.
        Headers are obtained as per `authenticate_and_get_headers(...)`, which typically
        depends on a short‑lived token produced by `generate_sp_token`.
      - Reads and (if needed) modifies the JSON payload to guarantee a `"model"` key
        (defaulting to `'gpt-4o'` when absent), then replaces the request body and
        updates `Content-Length` accordingly.
      - Modifies the url to ensure `'api-version'` parameter is present

    Args:
        workspace_url (str):
            Databricks workspace_url
        openai_apikey (str | None):
            Azure OpenAI API key. If omitted, pulled from `AZURE_OPENAI_API_KEY` or `OPENAI_API_KEY`.
        proxy_client_id (str | None):
            Client ID used for proxy authentication. If omitted,
            pulled from `PROXY_CLIENT_ID`.
        proxy_client_secret (str | None):
            Client secret used for proxy authentication. If omitted,
            pulled from `PROXY_CLIENT_SECRET`.
        api_version (str):
            [Deprecated] API version to enforce on outgoing requests via the `api-version` query parameter. 
            Defaults to `DEFAULT_API_VERSION`.
            .. deprecated:: 
            API version is no longer required since Azure OpenAI REST API v1.

    Returns:
        tuple[httpx.Client, httpx.AsyncClient]:
            `(custom_sync_client, custom_async_client)`—two clients preconfigured with
            the same request hook for authenticated, proxy‑compatible calls.

    Raises:
        KeyError: If any required environment variable (`AZURE_OPENAI_API_KEY`|`OPENAI_API_KEY`,
                  `PROXY_CLIENT_ID`, `PROXY_CLIENT_SECRET`) is missing when its
                  corresponding argument is `None`.
        json.JSONDecodeError: If the request body exists but is not valid JSON.
        UnicodeDecodeError: If request body bytes cannot be decoded as UTF‑8.
        Exception: Any error raised by `authenticate_and_get_headers(...)` (e.g.,
                   token acquisition failures).

    Examples:
        >>> from langchain_core.messages import HumanMessage
        >>> from langchain_openai import AzureChatOpenAI
        >>> from sungpt.agent_toolkit.utils.nginx_utils import http_client_factory 

        >>> api_version = "2025-04-01-preview" 
        >>> deployment_id = "gpt-4o"
        >>> proxy_port = "8110"
        >>> proxy_part = "openai-00010-1"
        >>> cluster_id = "0503-061117-o2rl78n9"

        >>> workspace_url = "https://suncorp-dev.cloud.databricks.com"
        >>> workspace_id = "106604617108771"

        >>> http_client, http_async_client = http_client_factory(
        ...     workspace_url,
        ...     openai_apikey=openai_apikey,
        ...     proxy_client_id=proxy_client_id,
        ...     proxy_client_secret=proxy_client_secret,
        ...     api_version=api_version,
        >>> )

        >>> model = AzureChatOpenAI(
        ...     # Do NOT include azure_deployment or azure_endpoint
        ...     api_version=api_version,
        ...     model=deployment_id,
        ...     # Pass the http clients here
        ...     http_client=http_client,
        ...     http_async_client=http_async_client,
        ...     api_key=openai_apikey,
        ...     # for chat completion, set base_url to ../{proxy_part}/openai/deployments/{model}
        ...     # for responses api, set base_url to ../{proxy_part}/openai
        ...     # note that base_url does not end with a slash (/)
        ...     base_url=f"{workspace_url}/driver-proxy-api/o/{workspace_id}/{cluster_id}/{proxy_port}/{proxy_part}/openai",
        ...     use_responses_api=True,
        ...     # Provide other necessary params
        >>> )
 
        >>> messages = [HumanMessage(content="Hello there ...")]
 
        >>> model.invoke(messages)
    """

    if openai_apikey is None:
        logger.debug("parameter `openai_apikey` is None; defaulting to os.environ['AZURE_OPENAI_API_KEY'] or os.environ['OPENAI_API_KEY'].")
        openai_apikey = os.environ.get('AZURE_OPENAI_API_KEY') if os.environ.get('AZURE_OPENAI_API_KEY') else os.environ.get('OPENAI_API_KEY')
    
    if proxy_client_id is None:
        logger.debug("parameter `proxy_client_id` is None; defaulting to os.environ['PROXY_CLIENT_SECRET'].")
        proxy_client_id = os.environ['PROXY_CLIENT_ID']

    if proxy_client_secret is None:
        logger.debug("parameter `proxy_client_secret` is None; defaulting to os.environ['PROXY_CLIENT_SECRET'].")
        proxy_client_secret = os.environ['PROXY_CLIENT_SECRET']

    if api_version is not None:
        warnings.warn(
            "The 'api_version' parameter is deprecated and will be removed in the next release."
            "Pass the api-version as part of your original request, if required."
            "If you are using Azure OpenAI REST API v1, api-version is not required.",
            category=FutureWarning,
            stacklevel=2
        )
        
    # --- 1. Define the nginx_request_hook function
    def _nginx_request_hook(request: httpx.Request):
        """
        This hook function inspects and modifies various parts of the outgoing request.
        """
        logger.info("--- Running Request Hook ---")

        # Inspect, log, and add the URL and HTTP parameters
        # ==================================================
        # Note: Params are part of the `request.url` object
        logger.debug(f"[URL] Original URL: {request.url}")
        
        logger.debug(f"[Params] Original query params: {request.url.params}")
        # if "api-version" not in request.url.params.keys(): 
        #     request.url.params["api-version"] = api_version
        #     logger.warn(f"[Params] upsert 'api-version' param to '{api_version}'")

        # Inspect, log, and update the headers
        # =========================================
        logger.debug(f"[Headers] Original headers: {request.headers}")
        auth_headers = authenticate_and_get_headers(openai_apikey, proxy_client_id, proxy_client_secret, workspace_url)
        for k, v in auth_headers.items():
            request.headers[k] = v
            logger.debug(f"[Headers] upsert '{k}' to '{v}'")

        # Inspect, log, and update the JSON payload
        # ===============================================
        # This is the most complex part because the body is a stream.
        # We need to read the stream, decode it, modify it, and then replace it.
        if not request.stream:
            logger.warn("[JSON] No request body to inspect or modify.")
        else:
            # Read the original body bytes from the stream
            # .read() consumes the stream, so we must replace it later.
            body_bytes = request.read()
            
            # Decode bytes to a string (assuming utf-8)
            body_str = body_bytes.decode('utf-8')
            logger.debug(f"[JSON] Original JSON payload (string): {body_str}")
            
            # Parse the JSON string into a Python dictionary
            data = json.loads(body_str)
            
            # Modify the data (making sure "model" is included)
            if "model" not in data.keys():
                data['model'] = 'gpt-4o'
                logger.warn(f"[JSON] Modified data dictionary: {data}")
            
            # Convert the modified dictionary back to a JSON string, then to bytes
            modified_body_bytes = json.dumps(data).encode('utf-8')
            
            # *** CRITICAL STEPS ***
            # Replace the stream with our new, modified content and update the
            # header with new Content-Length accordingly
            request.stream = httpx.ByteStream(modified_body_bytes)
            request.headers['Content-Length'] = str(len(modified_body_bytes))
            logger.debug(f"[JSON] Body replaced and Content-Length updated to {len(modified_body_bytes)}")

        logger.info("--- Finished Request Hook ---\n")

    # Token cache shared across all async requests from this client pair.
    # Tokens are valid for ~3600 s; refresh 60 s before expiry so concurrent
    # pages never race to fetch a new one mid-request.
    _cached_token: list = [None]
    _token_expiry: list = [0.0]

    def _get_proxy_token() -> str:
        """Return a cached proxy OAuth token, fetching a new one when near expiry."""
        if _cached_token[0] and time.time() < _token_expiry[0] - 60:
            logger.debug("Reusing cached proxy OAuth token")
            return _cached_token[0]
        logger.info("Fetching new proxy OAuth token")
        token = generate_sp_token(workspace_url, proxy_client_id, proxy_client_secret)
        _cached_token[0] = token
        _token_expiry[0] = time.time() + 3500  # tokens valid for 3600 s
        return token

    # Async version of the hook — httpx's AsyncClient awaits event hooks, so the
    # hook must be an async function.  The blocking _get_proxy_token() call is
    # offloaded to a thread via asyncio.to_thread() so it doesn't block the loop.
    async def _nginx_async_request_hook(request: httpx.Request):
        logger.info("--- Running Async Request Hook ---")

        logger.debug(f"[URL] Original URL: {request.url}")
        logger.debug(f"[Params] Original query params: {request.url.params}")

        logger.debug(f"[Headers] Original headers: {request.headers}")
        token = await asyncio.to_thread(_get_proxy_token)
        auth_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "api-key": openai_apikey,
        }
        for k, v in auth_headers.items():
            request.headers[k] = v
            logger.debug(f"[Headers] upsert '{k}' to '{v}'")

        if not request.stream:
            logger.warn("[JSON] No request body to inspect or modify.")
        else:
            body_bytes = request.read()
            body_str = body_bytes.decode('utf-8')
            logger.debug(f"[JSON] Original JSON payload (string): {body_str}")
            data = json.loads(body_str)
            if "model" not in data.keys():
                data['model'] = 'gpt-4o'
                logger.warn(f"[JSON] Modified data dictionary: {data}")
            modified_body_bytes = json.dumps(data).encode('utf-8')
            request.stream = httpx.ByteStream(modified_body_bytes)
            request.headers['Content-Length'] = str(len(modified_body_bytes))
            logger.debug(f"[JSON] Body replaced and Content-Length updated to {len(modified_body_bytes)}")

        logger.info("--- Finished Async Request Hook ---\n")

    # --- 2. Create your httpx clients with the hook ---
    # Sync client uses the sync hook; async client uses the async hook.
    # httpx's AsyncClient awaits event hooks — passing a sync hook causes
    # "TypeError: object NoneType can't be used in 'await' expression".
    custom_sync_client = httpx.Client(event_hooks={'request': [_nginx_request_hook]})
    custom_async_client = httpx.AsyncClient(event_hooks={'request': [_nginx_async_request_hook]})

    return (custom_sync_client, custom_async_client)