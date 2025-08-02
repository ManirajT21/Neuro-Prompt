from pyngrok import ngrok, conf
import os
import sys

print("=" * 60)
print("🔑 NGROK AUTHTOKEN SETUP")
print("=" * 60)

# Find the config path pyngrok uses
config_path = conf.DEFAULT_NGROK_PATH
token_set = False

if os.path.exists(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            if "authtoken:" in f.read():
                token_set = True
    except Exception as e:
        print(f"⚠️ Could not read ngrok config file: {e}")
        print("Continuing as if token is not set.")

if token_set:
    print("✅ ngrok authtoken is already set! You can use ngrok tunnels.")
else:
    print("Get your authtoken at: https://dashboard.ngrok.com/get-started/your-authtoken")
    token = input("Enter the token: ").strip()
    if not token:
        print("❌ No token entered. Exiting.")
        sys.exit(1)
    try:
        ngrok.set_auth_token(token)
        print("✅ ngrok authtoken set successfully! You can now use ngrok tunnels in your project.")
    except Exception as e:
        print(f"❌ Error setting authtoken: {e}")
        print("Please ensure pyngrok is installed and you are in the correct environment.")
        sys.exit(1)

print("=" * 60)
