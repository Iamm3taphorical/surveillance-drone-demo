// ControlsPanel.jsx
import React, { useState } from "react";

export default function ControlsPanel({ apiBase, token, onStart, onStop, onSelect, onZoomChange }) {
  const [source, setSource] = useState("usb:0");
  const [conf, setConf] = useState(0.25);
  const start = async () => {
    const res = await fetch(`${apiBase}/start?source=${encodeURIComponent(source)}`, { method: "POST", headers: { Authorization: `Bearer ${token}` }});
    onStart && onStart(res);
  };
  const stop = async () => {
    await fetch(`${apiBase}/stop`, { method: "POST", headers: { Authorization: `Bearer ${token}` }});
    onStop && onStop();
  };
  const changeConf = (v) => { setConf(v); /* set via REST? for simplicity send to /detect calls or POST an endpoint */ };

  return (
    <div className="p-3 bg-gray-800 text-white rounded space-y-2">
      <div>
        <label className="block text-sm">Source</label>
        <input value={source} onChange={(e)=>setSource(e.target.value)} className="w-full p-1 rounded text-black" />
        <div className="text-xs text-gray-400">Examples: usb:0 , rtsp:rtsp://192.168.x.x:554/stream , mjpeg:http://192.168.x.x:8080/video</div>
      </div>
      <div className="flex space-x-2">
        <button onClick={start} className="px-3 py-1 bg-green-600 rounded">Start</button>
        <button onClick={stop} className="px-3 py-1 bg-red-600 rounded">Stop</button>
      </div>
      <div>
        <label className="block text-sm">Confidence: {conf}</label>
        <input type="range" min="0.0" max="1.0" step="0.01" value={conf} onChange={(e)=>changeConf(parseFloat(e.target.value))}/>
      </div>
    </div>
  )
}
