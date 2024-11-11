import subprocess

# Open the requirements file
with open('requirements.txt', 'r') as file:
    requirements = file.readlines()

# Iterate through each package in the requirements file
for requirement in requirements:
    package = requirement.strip()  # Remove any extra whitespace or newline characters
    if package:  # Ensure it's not an empty line
        print(f"Installing {package}...")
        subprocess.run(["pip", "install", package], check=True)
