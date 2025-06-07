import React, { useRef, useEffect, useState } from "react";
import * as facemesh from "@tensorflow-models/face-landmarks-detection";
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import "./App.css";

const categories = {
  Home: [
    ["Yes", "No", "Please", "OK", "I", "You", "Want", "Stop"],
    ["It", "Will", "Feel", "Think", "Quick", "To", "Topics", "Actions"],
    ["Eat", "Drink", "Hungry", "Thirsty", "Apple", "Bread", "Water", "Juice"],
    ["Breakfast", "Lunch", "Dinner", "Snack", "More", "Less", "Finish", "Delicious"],
    ["Help", "Run", "Walk", "Sit", "Go", "Come", "Look", "Talk"],
    ["🎵", "⚽", "🏠", "✈️", "🔑", "🕰️", "💡", "🔒"]
  ],
  Action: [
    ["🏃", "🚶", "🗣️", "🤔", "😴", "💪", "🎧", "📝"],
    ["🙋", "🤝", "💃", "🕺", "🚗", "🏀", "⚽", "🏋️"],
    ["🏖️", "🏕️", "💼", "🎮", "💻", "📞", "💼", "🖥️"],
    ["🛀", "🛍️", "🎨", "🎤", "🎬", "🎯", "🎸", "🛠️"],
    ["🔧", "🔨", "🧩", "🎮", "📚", "📖", "📅", "🛠️"],
    ["🤽‍♂️", "🏃‍♀️", "⛹️‍♂️", "🤺", "🧘‍♂️", "🎻", "🎧", "🎼"]
  ],
  Emotions: [
    ["😊", "😢", "😠", "😱", "😅", "😎", "🥺", "😍"],
    ["🤔", "😜", "😤", "😇", "😒", "🥳", "😴", "😷"],
    ["🤩", "😜", "🤪", "😏", "😑", "🤮", "🤬", "😌"],
    ["🥺", "😬", "🥳", "😌", "😩", "🤗", "😭", "😡"],
    ["🥲", "🫣", "😏", "😣", "😓", "😕", "😶", "😝"],
    ["🙃", "🥱", "😓", "😔", "😙", "😊", "😖", "😌"]
  ],
  Words: [
    ["I", "You", "He", "She", "We", "They", "It", "Me"],
    ["This", "That", "Here", "There", "What", "When", "Where", "Why"],
    ["How", "Which", "Why", "Who", "Do", "Does", "Can", "Will"],
    ["Yes", "No", "Please", "Thanks", "Help", "Stop", "Go", "Come"],
    ["Want", "Need", "Like", "Love", "Hate", "Feel", "Think", "Say"],
    ["Start", "Finish", "Begin", "End", "Try", "Work", "Play", "Walk"],
    ["Look", "See", "Hear", "Feel", "Smell", "Touch", "Taste", "Speak"]
  ],
  Food: [
    ["🍕", "🍔", "🍝", "🍟", "🥗", "🍲", "🍣", "🌮"],
    ["🍚", "🍞", "🍜", "🍗", "🥩", "🐟", "🍳", "🧀"],
    ["🍎", "🍌", "🍊", "🥭", "🍇", "🍉", "🍍", "🍓"],
    ["🥕", "🥒", "🥬", "🍅", "🧅", "🧄", "🥔", "🌶️"],
    ["🍰", "🍨", "🥧", "🍪", "🍫", "🍩", "🧁", "🍫"],
    ["🧃", "🥤", "🍵", "☕", "🥛", "💧", "🍷", "🍺"]
  ]
};

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [eyeStatus, setEyeStatus] = useState("");
  const [currentCategory, setCurrentCategory] = useState("Home");
  const [filteredGrid, setFilteredGrid] = useState(null);
  const [selectionStage, setSelectionStage] = useState("column"); // could later be used for symbol-level selection
  const lastSelectionTime = useRef(0); // prevents rapid reselection
  const [isCooldown, setIsCooldown] = useState(false);



  const categoryGrid = categories[currentCategory];

  const eyeAspectRatio = (eye) => {
    const A = Math.hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1]);
    const B = Math.hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1]);
    const C = Math.hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1]);
    return (A + B) / (2.0 * C);
  };

  const filterColumns = (grid, columns) => {
    return grid.map(row => columns.map(colIdx => row[colIdx]));
  };

  const handleBlinkWithDirection = (direction) => {
    if (isCooldown) return; // Don't allow selection during cooldown
  
    const colsToShow = {
      "Looking Left": [0, 1, 2],
      "Looking Center": [3, 4],
      "Looking Right": [5, 6, 7]
    };
  
    const selectedCols = colsToShow[direction];
    if (!selectedCols) return;
  
    const selectedGrid = filterColumns(categoryGrid, selectedCols);
    setFilteredGrid(selectedGrid);
    setIsCooldown(true); // start cooldown
  
    // Wait 10 seconds before resetting grid and resuming detection
    setTimeout(() => {
      setFilteredGrid(null);
      setIsCooldown(false);
    }, 30000); // 10 seconds
  };
  
  

  const detect = async (net) => {
    if (webcamRef.current && webcamRef.current.video.readyState === 4) {
      const video = webcamRef.current.video;
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const face = await net.estimateFaces({ input: video, returnTensors: false, flipHorizontal: false });
      if (!canvasRef.current) return;
      const ctx = canvasRef.current.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, videoWidth, videoHeight);

      if (face.length > 0) {
        const keypoints = face[0].scaledMesh;
        const leftPupil = keypoints[468];
        const rightPupil = keypoints[473];

        drawCircle(ctx, leftPupil, "green");
        drawCircle(ctx, rightPupil, "green");

        const avgX = (leftPupil[0] + rightPupil[0]) / 2;
        const thresholdLeft = videoWidth * 0.35;
        const thresholdRight = videoWidth * 0.65;

        const leftEye = [keypoints[362], keypoints[385], keypoints[387], keypoints[263], keypoints[373], keypoints[380]];
        const rightEye = [keypoints[33], keypoints[160], keypoints[158], keypoints[133], keypoints[153], keypoints[144]];

        const earLeft = eyeAspectRatio(leftEye);
        const earRight = eyeAspectRatio(rightEye);
        const ear = (earLeft + earRight) / 2;

        let gazeDirection = "";

        if (avgX < thresholdLeft) gazeDirection = "Looking Right";
        else if (avgX > thresholdRight) gazeDirection = "Looking Left";
        else gazeDirection = "Looking Center";

        if (ear < 0.2) {
          setEyeStatus(`${gazeDirection} + Blink`);
          handleBlinkWithDirection(gazeDirection);
        } else {
          setEyeStatus(gazeDirection);
        }
      }
    }
  };

  const drawCircle = (ctx, point, color) => {
    ctx.beginPath();
    ctx.arc(point[0], point[1], 4, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  };

  const runFacemesh = async () => {
    const net = await facemesh.load(facemesh.SupportedPackages.mediapipeFacemesh, { maxFaces: 1 });
    setInterval(() => detect(net), 100);
  };

  useEffect(() => {
    const load = async () => {
      await tf.setBackend('webgl');
      await tf.ready();
      runFacemesh();
    };
    load();
  }, []);

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden", fontFamily: "Arial" }}>
      <Webcam ref={webcamRef} style={{ position: "absolute", left: 0, top: 0, zIndex: 9, width: 540, height: 480, opacity: 0 }} />
      <canvas ref={canvasRef} style={{ position: "absolute", left: 0, top: 0, zIndex: 9, width: 400, height: 480 }} />

      <div style={{ width: "80px", background: "#333", padding: "10px", display: "flex", flexDirection: "column", gap: "10px" }}>
        {Object.keys(categories).map((cat) => (
          <button key={cat} onClick={() => setCurrentCategory(cat)} style={{ backgroundColor: "lightblue", fontSize: "16px", padding: "10px", borderRadius: "8px" }}>
            {cat}
          </button>
        ))}
      </div>

      <div style={{ flex: 1, padding: "10px", overflow: "auto" }}>
        <div style={{ width: "100%", border: "2px solid black", padding: "10px", backgroundColor: "white", fontSize: "24px", marginBottom: "10px", height: "70px", display: "flex", justifyContent: "center", alignItems: "center" }}>
          Eye Tracker Communication System
        </div>

        <div style={{ width: "100%", textAlign: "center", fontSize: "22px", padding: "10px", marginBottom: "10px", backgroundColor: eyeStatus.includes("Blink") ? "#ffdddd" : "#ddffdd", borderRadius: "10px", fontWeight: "bold" }}>
          {eyeStatus}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: "10px" }}>
          {(filteredGrid || categoryGrid).flat().map((item, idx) => (
            <div key={idx} style={{
              backgroundColor: "lightblue",
              padding: "20px 10px",
              fontSize: "18px",
              borderRadius: "8px",
              textAlign: "center",
              height: "60px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center"
            }}>
              {item}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;

