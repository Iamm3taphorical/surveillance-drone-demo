// VideoPlayer.jsx
import React, { useEffect, useRef, useState } from "react";

export default function VideoPlayer({ token, streamUrl, onOverlayClick, onROIChange }) {
  const imgRef = useRef();
  const canvasRef = useRef();
  const [boxes, setBoxes] = useState([]);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!token) return;
    const url = `${(window.location.protocol === 'https:' ? 'wss' : 'ws')}://${window.location.host.replace(/:\\d+$/, ':8000')}/ws?token=${token}`;
    const ws = new WebSocket(url);
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        setBoxes(data.detections || []);
      } catch (e) { console.error(e); }
    };
    wsRef.current = ws;
    return () => { ws.close(); };
  }, [token]);

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0,0,canvas.width, canvas.height);
    boxes.forEach(b => {
      const x = (b.xmin / img.naturalWidth) * canvas.width;
      const y = (b.ymin / img.naturalHeight) * canvas.height;
      const w = ((b.xmax - b.xmin) / img.naturalWidth) * canvas.width;
      const h = ((b.ymax - b.ymin) / img.naturalHeight) * canvas.height;
      ctx.strokeStyle = "lime";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      ctx.font = "14px sans-serif";
      ctx.fillStyle = "lime";
      ctx.fillText(`#${b.id || "-"} ${b.name} ${(b.conf||0).toFixed(2)}`, x + 4, y + 14);
    });
  }, [boxes]);

  const handleClick = (e) => {
    const rect = e.target.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const clicked = boxes.find(b => {
      const x = (b.xmin / e.target.naturalWidth) * rect.width;
      const y = (b.ymin / e.target.naturalHeight) * rect.height;
      const w = ((b.xmax - b.xmin) / e.target.naturalWidth) * rect.width;
      const h = ((b.ymax - b.ymin) / e.target.naturalHeight) * rect.height;
      return cx >= x && cx <= x+w && cy >= y && cy <= y+h;
    });
    if (clicked && onOverlayClick) onOverlayClick(clicked);
  };

  return (
    <div className="relative">
      <img ref={imgRef} src={streamUrl} alt="stream" onClick={handleClick} className="w-full h-auto" />
      <canvas ref={canvasRef} className="absolute top-0 left-0 pointer-events-none" />
    </div>
  )
}
