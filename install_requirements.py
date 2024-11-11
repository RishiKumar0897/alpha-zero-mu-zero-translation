import subprocess

with open('requirements.txt', 'r') as file:
    packages = file.readlines()


for package in packages:
    package = package.strip()

    if package:
        try:
            print(f"Intalling {package}...")
            subprocess.check_call(['pip', 'install', package])
        except subprocess.CalledProcessError:
            print("Failed ot install {package}. Skipping...")