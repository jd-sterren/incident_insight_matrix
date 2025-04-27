import os
import inc.functions as fn
from inc.credential_manager import inject_decrypted_env, get_passphrase  # << New import

# Obtain the passphrase from hidden file or user input
passphrase = get_passphrase()

# Inject decrypted environment variables
inject_decrypted_env(environment="prod", passphrase=passphrase)

if __name__ == "__main__":
    # Call the function to collect GPS data
    fn.collect_gps_data()