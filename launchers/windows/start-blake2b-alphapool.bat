@echo off
REM X-Miner - Bitcoin Knots BLAKE2b (RDTS) on AlphaPool. No X-Miner dev fee on blake2b.
REM Edit WALLET (your RDTS payout address) and WORKER below, then double-click this file.
REM AlphaPool's share difficulty is sized for ASICs: a GPU finds a share every few hours or days (lottery mining).
cd /d "%~dp0"

set WALLET=YOUR_RDTS_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=us1.alphapool.tech:5555

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Measured examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%% --gpu-voffset -50   (+3.5%% at the same power)
REM   RTX 3060:   set TUNING=--gpu-mclock 810   (same speed, 15 W less)
set TUNING=

:loop
x-miner.exe --algorithm blake2b --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
