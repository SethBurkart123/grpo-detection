#!/usr/bin/env python3

import sys
import webbrowser

from src.claude.anthropic_oauth import AnthropicOAuth


class ClaudeClient:
    def __init__(self, auto_open_browser: bool = True):
        self.oauth = AnthropicOAuth()
        self.auto_open_browser = auto_open_browser
    
    def _ensure_authenticated(self) -> bool:
        """Ensure user is authenticated, prompting for login if needed."""
        if self.oauth.is_authenticated():
            return True
        
        print("Not authenticated. Starting login process...")
        return self._do_login()
    
    def _do_login(self) -> bool:
        """Handle the login flow."""
        try:
            auth_data = self.oauth.authorize()
            
            print("\nPlease authorize the application:")
            print(f"Authorization URL: {auth_data['url']}")
            
            if self.auto_open_browser:
                print("\nOpening browser...")
                try:
                    webbrowser.open(auth_data['url'])
                except Exception as e:
                    print(f"Could not open browser: {e}")
                    print("Please manually open the URL above")
            
            print("\nAfter authorizing, copy the code from the page.")
            
            code = input("Enter authorization code: ").strip()
            if not code:
                print("No code provided")
                return False
            
            print("Exchanging code for tokens...")
            self.oauth.exchange_code(code, auth_data['verifier'])
            print("Login successful!")
            return True
            
        except KeyboardInterrupt:
            print("\nLogin cancelled")
            return False
        except Exception as e:
            print(f"Login failed: {e}")
            return False
    
    def send_message(self, content: str, model: str = "claude-sonnet-4-20250514", max_tokens: int = 1024, system_prompt: str = "You are a helpful assistant.", history: list = None):
        """
        Send a message to Claude, auto-authenticating if needed.
        
        Returns:
            str: response_text (if history=None)
            tuple: (response_text, updated_history) (if history was passed)
        """
        if not self._ensure_authenticated():
            if history is not None:
                return None, history
            else:
                return None
        
        try:
            if history is not None:
                # Use conversation history
                current_history = history.copy()
                current_history.append({'role': 'user', 'content': content})
                
                response = self.oauth.send_message(
                    messages=current_history,
                    model=model,
                    max_tokens=max_tokens,
                    custom_system_prompt=system_prompt
                )
            else:
                # Send single message (original behavior)
                response = self.oauth.send_message(content, model, max_tokens, system_prompt)
            
            if 'content' in response and response['content']:
                assistant_response = response['content'][0].get('text', 'No text in response')
                
                if history is not None:
                    # Add both user message and assistant response to history
                    updated_history = history.copy()
                    updated_history.append({'role': 'user', 'content': content})
                    updated_history.append({'role': 'assistant', 'content': assistant_response})
                    return assistant_response, updated_history
                else:
                    return assistant_response
            else:
                print("Unexpected response format:")
                print(response)
                if history is not None:
                    return None, history
                else:
                    return None
                
        except Exception as e:
            print(f"Error sending message: {e}")
            if history is not None:
                return None, history
            else:
                return None
    
    def chat(self, content: str, model: str = "claude-sonnet-4-20250514", max_tokens: int = 1024, system_prompt: str = "You are a helpful assistant.", history: list = None):
        """Alias for send_message for a more natural interface."""
        return self.send_message(content, model, max_tokens, system_prompt, history)
    
    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        return self.oauth.is_authenticated()
    
    def logout(self) -> None:
        """Clear authentication."""
        self.oauth.clear_auth()
        print("Logged out successfully")


def claude_chat(message: str, model: str = "claude-sonnet-4-20250514", max_tokens: int = 1024, system_prompt: str = "You are a helpful assistant.", history: list = None, auto_open_browser: bool = True):
    """
    Simple function to chat with Claude. Auto-handles authentication.
    
    Args:
        message: The message to send to Claude
        model: The model to use (default: claude-sonnet-4-20250514)
        max_tokens: Maximum tokens for response (default: 1024)
        system_prompt: Custom system prompt (default: "You are a helpful assistant.")
        history: Optional conversation history (default: None for single message)
        auto_open_browser: Whether to auto-open browser for auth (default: True)
    
    Returns:
        str: response_text (if history=None)
        tuple: (response_text, updated_history) (if history was passed)
    """
    client = ClaudeClient(auto_open_browser=auto_open_browser)
    return client.chat(message, model, max_tokens, system_prompt, history)


def main():
    """Interactive chat mode."""
    client = ClaudeClient()
    
    print("Claude Chat Client")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'logout' to clear authentication")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
            elif user_input.lower() == 'logout':
                client.logout()
                continue
            elif not user_input:
                continue
            
            response = client.chat(user_input)
            if response:
                print(f"\nClaude: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main() 