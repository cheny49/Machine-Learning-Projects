import tarfile
import os

os.chdir(r"C:\Users\cheny\Downloads")
tfile = tarfile.open("cifar-10-python.tar.gz", "r:gz")
tfile.extractall(".")

print("you're ready to go!")
