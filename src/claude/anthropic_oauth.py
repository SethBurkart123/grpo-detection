#!/usr/bin/env python3

import argparse
import base64
import hashlib
import json
import os
import secrets
import sys
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Dict, Optional, Union

import requests

class AnthropicOAuth:
    
    CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    AUTH_URL = "https://claude.ai/oauth/authorize"
    TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
    REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
    API_URL = "https://api.anthropic.com/v1/messages"
    SCOPES = "org:create_api_key user:profile user:inference"
    
    def __init__(self):
        if os.name == 'nt':
            self.config_dir = Path(os.environ.get('APPDATA', '~')) / 'anthropic-oauth'
        else:
            self.config_dir = Path.home() / '.local' / 'share' / 'anthropic-oauth'
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.auth_file = self.config_dir / 'auth.json'
        
        try:
            os.chmod(self.config_dir, 0o700)
        except OSError:
            pass

    def _generate_pkce(self) -> Dict[str, str]:
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        
        return {
            'verifier': code_verifier,
            'challenge': code_challenge
        }

    def authorize(self) -> Dict[str, str]:
        pkce = self._generate_pkce()
        
        params = {
            'code': 'true',
            'client_id': self.CLIENT_ID,
            'response_type': 'code',
            'redirect_uri': self.REDIRECT_URI,
            'scope': self.SCOPES,
            'code_challenge': pkce['challenge'],
            'code_challenge_method': 'S256',
            'state': pkce['verifier']
        }
        
        auth_url = f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"
        
        return {
            'url': auth_url,
            'verifier': pkce['verifier']
        }

    def exchange_code(self, code: str, verifier: str) -> Dict[str, Union[str, int]]:
        code_parts = code.split('#')
        auth_code = code_parts[0]
        state = code_parts[1] if len(code_parts) > 1 else None
        
        payload = {
            'code': auth_code,
            'state': state,
            'grant_type': 'authorization_code',
            'client_id': self.CLIENT_ID,
            'redirect_uri': self.REDIRECT_URI,
            'code_verifier': verifier
        }
        
        response = requests.post(
            self.TOKEN_URL,
            headers={'Content-Type': 'application/json'},
            json=payload
        )
        
        if not response.ok:
            raise Exception(f"Token exchange failed: {response.status_code} {response.text}")
        
        token_data = response.json()
        
        auth_info = {
            'type': 'oauth',
            'access_token': token_data['access_token'],
            'refresh_token': token_data['refresh_token'],
            'expires_at': int(time.time()) + token_data['expires_in']
        }
        
        self._save_auth(auth_info)
        return auth_info

    def refresh_access_token(self) -> Optional[str]:
        auth_info = self._load_auth()
        if not auth_info or auth_info.get('type') != 'oauth':
            return None
        
        payload = {
            'grant_type': 'refresh_token',
            'refresh_token': auth_info['refresh_token'],
            'client_id': self.CLIENT_ID
        }
        
        response = requests.post(
            self.TOKEN_URL,
            headers={'Content-Type': 'application/json'},
            json=payload
        )
        
        if not response.ok:
            return None
        
        token_data = response.json()
        
        auth_info.update({
            'access_token': token_data['access_token'],
            'refresh_token': token_data['refresh_token'],
            'expires_at': int(time.time()) + token_data['expires_in']
        })
        
        self._save_auth(auth_info)
        return token_data['access_token']

    def get_access_token(self) -> Optional[str]:
        auth_info = self._load_auth()
        if not auth_info or auth_info.get('type') != 'oauth':
            return None
        
        if auth_info['expires_at'] > int(time.time()) + 300:
            return auth_info['access_token']
        
        return self.refresh_access_token()

    def _save_auth(self, auth_info: Dict[str, Union[str, int]]) -> None:
        with open(self.auth_file, 'w') as f:
            json.dump(auth_info, f, indent=2)
        
        try:
            os.chmod(self.auth_file, 0o600)
        except OSError:
            pass

    def _load_auth(self) -> Optional[Dict[str, Union[str, int]]]:
        if not self.auth_file.exists():
            return None
        
        try:
            with open(self.auth_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def clear_auth(self) -> None:
        if self.auth_file.exists():
            self.auth_file.unlink()

    def is_authenticated(self) -> bool:
        return self.get_access_token() is not None

    def send_message(self, content: str = None, model: str = "claude-sonnet-4-20250514", max_tokens: int = 1024, custom_system_prompt: str = "You are an expert poet.", messages: list = None) -> Dict:
        access_token = self.get_access_token()
        if not access_token:
            raise Exception("Not authenticated")
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}',
            'anthropic-version': '2023-06-01',
            'anthropic-beta': 'oauth-2025-04-20'
        }
        
        # Use provided messages or create single message from content
        if messages is not None:
            message_list = messages
        elif content is not None:
            message_list = [{'role': 'user', 'content': content}]
        else:
            raise ValueError("Either content or messages must be provided")
        
        payload = {
            'model': model,
            'max_tokens': max_tokens,
            'messages': message_list,
            "system": [
                {
                  "type": "text",
                  "text": "You are Claude Code, Anthropic's official CLI for Claude."
                },
                {
                    "type": "text",
                    "text": custom_system_prompt
                }
            ],
        }
        
        response = requests.post(self.API_URL, headers=headers, json=payload)
        
        if not response.ok:
            raise Exception(f"API request failed: {response.status_code} {response.text}")
        
        return response.json()


def main():
    parser = argparse.ArgumentParser(description="Anthropic OAuth CLI")
    
    parser.add_argument(
        'command',
        choices=['login', 'logout', 'status', 'test'],
        help='Command to run'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Don\'t open browser for login'
    )
    
    args = parser.parse_args()
    
    oauth = AnthropicOAuth()
    
    try:
        if args.command == 'login':
            auth_data = oauth.authorize()
            
            print("Starting OAuth login...")
            print("\nAuthorization URL:")
            print(auth_data['url'])
            
            if not args.no_browser:
                print("\nOpening browser...")
                try:
                    webbrowser.open(auth_data['url'])
                except Exception as e:
                    print(f"Could not open browser: {e}")
                    print("Please manually open the URL above")
            
            print("\nAfter authorizing, copy the code from the page.")
            
            try:
                code = input("\nEnter authorization code: ").strip()
                if not code:
                    print("No code provided")
                    sys.exit(1)
                
                print("Exchanging code for tokens...")
                oauth.exchange_code(code, auth_data['verifier'])
                print("Login successful")
                
            except KeyboardInterrupt:
                print("\nLogin cancelled")
                sys.exit(1)
            except Exception as e:
                print(f"Login failed: {e}")
                sys.exit(1)
        
        elif args.command == 'logout':
            oauth.clear_auth()
            print("Logged out successfully")
        
        elif args.command == 'status':
            if oauth.is_authenticated():
                auth_info = oauth._load_auth()
                expires_at = auth_info.get('expires_at', 0)
                expires_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expires_at))
                print("Authenticated")
                print(f"Token expires: {expires_str}")
            else:
                print("Not authenticated")
                print("Run 'login' command to authenticate")
        
        elif args.command == 'test':
            if not oauth.is_authenticated():
                print("Not authenticated. Please run 'login' command first.")
                sys.exit(1)
            
            print("Sending test message to Claude...")
            try:
                response = oauth.send_message("Write a beautiful poem about the beauty of the world.", custom_system_prompt="You are an expert poet.")
                
                if 'content' in response and response['content']:
                    message = response['content'][0].get('text', 'No text in response')
                    print(f"Claude responded: {message}")
                else:
                    print("API call successful but unexpected response format:")
                    print(json.dumps(response, indent=2))
                
            except Exception as e:
                print(f"Test failed: {e}")
                sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
