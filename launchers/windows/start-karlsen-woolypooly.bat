@echo off
REM X-Miner - Karlsen (KLS, KarlsenHashV2) on WoolyPooly (your karlsen:... address)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: pool.woolypooly.com:3132
cd /d "%~dp0"

set WALLET=YOUR_KLS_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=pool.woolypooly.com:3132

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%%
REM   KarlsenHashV2 reads the same 4.5 GB dataset as FishHash: a memory overclock helps.
set TUNING=

:loop
x-miner.exe --algorithm karlsenhashv2 --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
