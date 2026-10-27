' start-silent.vbs - klik dua kali untuk menjalankan layanan TANPA jendela apa pun.
' Bila .venv belum ada, start.bat dijalankan dengan jendela agar proses setup terlihat.
Option Explicit
Dim sh, fso, dir, bat, rc
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
bat = """" & dir & "\start.bat"""
If fso.FileExists(dir & "\.venv\Scripts\python.exe") Then
  rc = sh.Run(bat & " /silent", 0, True)
  If rc <> 0 Then
    MsgBox "Layanan gagal dijalankan. Lihat folder logs di:" & vbCrLf & dir, vbExclamation, "Portal Simulator dan ML"
  End If
Else
  sh.Run bat, 1, False
End If
