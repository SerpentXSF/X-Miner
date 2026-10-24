@echo off
REM X-Miner - Bitcoin Knots BLAKE2b (RDTS) on AlphaPool. No X-Miner dev fee on blake2b.
REM Edit WALLET (your RDTS payout address) and WORKER below, then double-click this file.
REM AlphaPool's share difficulty is sized for ASICs: a GPU finds a share every few hours or days (lottery mining).
cd /d "%~dp0"

set WALLET=YOUR_RDTS_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=us1.alphapool.tech:5555

:loop
x-miner.exe --algorithm blake2b --pool %POOL% --wallet %WALLET% --worker %WORKER%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
