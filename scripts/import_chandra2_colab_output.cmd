@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0import_chandra2_colab_output.ps1" %*
