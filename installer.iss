; Inno Setup script for GuitarViz
; Requires Inno Setup 6: https://jrsoftware.org/isdl.php

#define MyAppName      "GuitarViz"
#define MyAppVersion   "1.0"
#define MyAppPublisher "GuitarViz"
#define MyAppExeName   "GuitarViz.exe"
#define SourceDir      "dist\GuitarViz"

[Setup]
AppId={{A3F8B2C1-4D7E-4F9A-B123-567890ABCDEF}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppContact=https://github.com/chenisan/guitarvx

; 安裝路徑
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}

; 解除安裝顯示設定（出現在「控制台 → 程式和功能」）
UninstallDisplayName={#MyAppName} {#MyAppVersion}
UninstallDisplayIcon={app}\{#MyAppExeName}

; 需要管理員權限才能正確登錄到控制台
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64
MinVersion=10.0

; 輸出
OutputDir=Output
OutputBaseFilename=GuitarViz_Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[CustomMessages]
english.SmartScreenNote=Note: Windows SmartScreen may show a warning because this app is not yet code-signed. Click "More info" then "Run anyway" to proceed.
chinesesimplified.SmartScreenNote=注意：因本程式尚未購買程式碼簽章憑證，Windows SmartScreen 可能顯示警告。請點選「更多資訊」→「仍要執行」即可正常安裝。

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}";             Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}";   Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}";     Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Code]
procedure InitializeWizard;
begin
  MsgBox(CustomMessage('SmartScreenNote'), mbInformation, MB_OK);
end;

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 解除安裝時一併清除整個安裝資料夾（包含 PyInstaller 產生的所有 DLL）
Type: filesandordirs; Name: "{app}"
