@echo off
cd /d "D:\code\python\PyNG-Tuber"
start "" py -3.11 main.py

start "" "D:\BlenderFoundation\GooEngine\goo-engine-experimental-4_3\blender-launcher.exe" "D:\code\python\PyNG-Tuber\VTuber GooEngine Fem.blend"

cd /d "D:\obs-studio\bin\64bit"
start "" "obs64.exe"