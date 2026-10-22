@echo off
REM X-Miner - Pearl (PRL) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: prl-us.kryptex.network:7048!prl.kryptex.network:7048
cd /d "%~dp0"

set WALLET=YOUR_KRYPTEX_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=prl-us.kryptex.network:7048!prl.kryptex.network:7048

:loop
x-miner.exe --algorithm pearlhash --pool %POOL% --wallet %WALLET% --worker %WORKER%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
