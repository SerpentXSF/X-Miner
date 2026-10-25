@echo off
REM X-Miner - Quantus (QTC) on Kryptex (payouts in BTC/USDT to your Kryptex account; use your krx... username)
REM Edit WALLET and WORKER below, then double-click this file.
REM Pools: qtc-us.kryptex.network:7049!qtc.kryptex.network:7049
cd /d "%~dp0"

set WALLET=YOUR_KRYPTEX_ADDRESS
set WORKER=%COMPUTERNAME%
set POOL=qtc-us.kryptex.network:7049!qtc.kryptex.network:7049

:loop
x-miner.exe --algorithm quantus --pool %POOL% --wallet %WALLET% --worker %WORKER%
echo X-Miner exited (code %ERRORLEVEL%). Restarting in 10 seconds... (close this window to stop)
timeout /t 10 >nul
goto loop
