const { app, BrowserWindow } = require('electron');
const path = require('path');
const ffmpegPath = require('@ffmpeg-installer/ffmpeg').path;
const ffmpeg = require('@ffmpeg-installer/ffmpeg');
console.log('FFmpeg path:', ffmpegPath);


let mainWindow;  

const createWindow = () => {
    const iconPath = process.platform === 'darwin'
        ? path.join(__dirname, './src/assets/images/holoicns.icns') 
        : process.platform === 'linux'
        ? path.join(__dirname, './src/assets/images/holopng.png') 
        : path.join(__dirname, './src/assets/images/holoico.ico'); 

    mainWindow = new BrowserWindow({  
        width: 1200,
        height: 700,
        icon: iconPath,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js')
        }
    });
    mainWindow.setMenuBarVisibility(false)
    mainWindow.loadFile('./src/HomeScreen.html');  
  
};

app.whenReady().then(() => {
    createWindow();  

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow(); 
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();  
});
