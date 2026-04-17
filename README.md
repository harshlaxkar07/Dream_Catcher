# Dream_Catcher AI - Local Setup Guide

This guide will help you run the **Dream_Catcher AI** project on your own computer using **VS Code**.

## 1. Prerequisites (What you need to install)

Before you start, make sure you have these two things installed on your computer:

1.  **Node.js**: This is the engine that runs the project.
    *   Download it from: [https://nodejs.org/](https://nodejs.org/) (Choose the **LTS** version).
2.  **VS Code**: The code editor.
    *   Download it from: [https://code.visualstudio.com/](https://code.visualstudio.com/)

---

## 2. Step-by-Step Setup

### Step 1: Download the Project
Download the project files to your computer and extract them into a folder (e.g., `C:\Projects\Dream_Catcher`).

### Step 2: Open the Project in VS Code
1.  Open **VS Code**.
2.  Go to `File` -> `Open Folder...`
3.  Select the folder where you extracted the project files.

### Step 3: Open the Terminal
In VS Code, go to the top menu and select `Terminal` -> `New Terminal`. A small window will open at the bottom of VS Code.

### Step 4: Install Dependencies
In the terminal window, type the following command and press **Enter**:
```bash
npm install
```
*Wait for it to finish. This downloads all the necessary libraries (like React and MediaPipe).*

### Step 5: Run the Project
Once the installation is finished, type this command and press **Enter**:
```bash
npm run dev
```

### Step 6: Open in Browser
After running the command, you will see a link in the terminal that looks like this:
`http://localhost:3000`

**Hold Ctrl and click the link**, or copy and paste it into your web browser (Chrome or Edge recommended).

---

## 3. Troubleshooting

*   **Camera not working?** Make sure you grant camera permissions to the browser when prompted.
*   **Command not found?** Ensure you have installed Node.js correctly and restarted VS Code.
*   **Slow performance?** This app uses your computer's GPU for AI detection. Ensure your browser has "Hardware Acceleration" enabled in its settings.

---

## 4. Project Structure
*   `src/App.tsx`: The main logic and UI of the application.
*   `package.json`: List of libraries used by the project.
*   `public/`: Folder for static assets (if any).
# Dream_Catcher
