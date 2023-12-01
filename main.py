import subprocess
import os




os.chdir("home/THESIs/")

subprocess.run(["python", "prog_th4.py"])
subprocess.run(["python", "prog_th3.py"])
subprocess.run(["python", "prog_th2.py"])
subprocess.run(["python", "load_data.py"])
