@echo off
REM X-Miner - Qubitcoin (QTC, QHash) on LuckyPool (your Qubitcoin address: bc1q..., 3... or 1...)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: qtc-us-east.lproute.com:8610!qtc-eu.lproute.com:8610
cd /d "%~dp0"

set WALLET=YOUR_QUBITCOIN_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=qtc-us-east.lproute.com:8610!qtc-eu.lproute.com:8610

REM Optional GPU tuning, one value per GPU or a comma list (see README, "GPU tuning"). Examples:
REM   RX 9070 XT: set TUNING=--gpu-plimit 70%%
REM   QHash is SHA-256 plus a small 16-qubit circuit: compute bound, runs at the power limit.
set TUNING=

:loop
x-miner.exe --algorithm qhash --pool %POOL% --wallet %WALLET% --worker %WORKER% %TUNING%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
