# Dream Catcher

A real-time drowsiness detection system that runs entirely in your browser. It watches eye closure, head pose and yawning through the webcam, scores fatigue continuously, and raises an alarm before a driver or operator nods off.

Face tracking runs locally through **MediaPipe Vision** compiled to WebAssembly — the video never leaves the machine.

---

## Highlights

| | |
|---|---|
| **Runs in the browser** | MediaPipe's face landmarker runs on WebAssembly, using the GPU where available |
| **Nothing is uploaded** | Frames are analysed locally; no video is sent anywhere |
| **Three fatigue signals** | Eye closure, head tilt and yawning are combined into one score |
| **Audible alarm** | Fires when the fatigue score crosses the threshold, and stands down when eyes reopen |
| **Adjustable sensitivity** | Tune how readily the system escalates to an alert |
| **Live charts** | Fatigue over time and a breakdown of drowsiness events, drawn with Recharts |
| **Session history** | Events are logged, reviewable and exportable |
| **Responsive dashboard** | Works from a phone screen up to a full monitor |

---

## How detection works

```
Webcam frame
   │
   ├─ 1. Landmark   MediaPipe FaceLandmarker returns face geometry and blendshapes
   ├─ 2. Read       eye closure, jaw opening and head transform are pulled out
   ├─ 3. Score      the three signals are weighted and scaled by your sensitivity setting
   └─ 4. Act        crossing the threshold logs an event and sounds the alarm
```

The three contributing signals:

| Signal | Read from | What it catches |
|---|---|---|
| **Eye closure** | Eye blendshape scores | Long blinks and micro-sleeps |
| **Head tilt** | The face transformation matrix | The head dropping forward — nodding off |
| **Yawning** | The `jawOpen` blendshape | Yawns, a leading indicator of fatigue |

The alarm holds while the eyes remain closed and stands down once they reopen, so a single long blink does not leave it ringing.

---

## Tech stack

**Framework** React 19 · TypeScript · Vite 6
**Vision** MediaPipe Tasks Vision (WebAssembly)
**Charts** Recharts
**Motion** Motion (Framer Motion)
**Styling** Tailwind CSS 4
**Icons** Lucide

---

## Getting started

### Prerequisites

- [Node.js](https://nodejs.org) 18 or newer (the LTS build is fine)
- A browser with webcam access — Chrome or Edge recommended for GPU acceleration

### 1. Install

```bash
git clone https://github.com/harshlaxkar07/Dream_Catcher.git
cd Dream_Catcher
npm install
```

### 2. Run

```bash
npm run dev
```

Open `http://localhost:3000` and grant camera permission when the browser asks.

### 3. Build for production

```bash
npm run build      # outputs to dist/
npm run preview    # serve the built bundle locally
```

---

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Development server on port 3000, listening on all interfaces |
| `npm run build` | Production bundle into `dist/` |
| `npm run preview` | Serve the production bundle locally |
| `npm run lint` | Type-check with `tsc --noEmit` |
| `npm run clean` | Remove `dist/` |

---

## Configuration

An optional `.env` file, modelled on `.env.example`:

```ini
GEMINI_API_KEY="your-key"
APP_URL="http://localhost:3000"
```

---

## Using the dashboard

- **Sensitivity** — how readily the system escalates. Higher reacts sooner.
- **Sound** — mute or unmute the alarm from the header.
- **Pause** — stop analysis without giving up the camera.
- **Zoom** — scale the camera preview to frame the face well.
- **History** — the session's events, reviewable and exportable.
- **Charts** — fatigue over time and events grouped by kind.

For the best results: sit so your whole face is in frame, keep the light in front of you rather than behind, and leave hardware acceleration enabled in your browser so the model runs on the GPU.

---

## Project structure

```
Dream_Catcher/
├── index.html                          Vite entry point
├── vite.config.ts                      Build configuration
├── tsconfig.json                       TypeScript configuration
├── metadata.json                       App name, description and camera permission
├── src/
│   ├── App.tsx                         Detection loop, scoring, alarm and dashboard
│   ├── main.tsx                        React root
│   └── index.css                       Tailwind entry
└── public/
    ├── face_landmarker.task            The MediaPipe face landmark model
    └── wasm/                           MediaPipe WebAssembly runtime
```

The model and the WebAssembly runtime are served from `public/`, so the app runs without reaching out to a CDN at load time.
