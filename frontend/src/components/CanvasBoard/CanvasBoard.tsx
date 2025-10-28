import { useEffect, useRef, useState, useCallback } from 'react';
import { Overlay } from './Overlay';

export type Tool = 'brush' | 'eraser' | 'pan';

interface CanvasBoardProps {
  width: number;
  height: number;
  zoom: number;
  brushSize: number;
  tool: Tool;
  onExport: (dataURL: string) => void;
  onClear: () => void;
  onImageLoad?: (dataURL: string) => void;
  loadImageURL?: string;
}

export function CanvasBoard({
  width,
  height,
  zoom,
  brushSize,
  tool,
  onExport,
  onClear,
  onImageLoad,
  loadImageURL,
}: CanvasBoardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const panStartRef = useRef({ x: 0, y: 0, offsetX: 0, offsetY: 0 });
  const spaceKeyRef = useRef(false);

  // Initialize canvas with HiDPI support
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctxRef.current = ctx;
  }, [width, height]);

  // Handle keyboard for Space + drag pan
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !spaceKeyRef.current) {
        e.preventDefault();
        spaceKeyRef.current = true;
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        e.preventDefault();
        spaceKeyRef.current = false;
        setIsPanning(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  // Export canvas to PNG
  const exportToPNG = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return '';

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return '';

    tempCtx.fillStyle = '#ffffff';
    tempCtx.fillRect(0, 0, width, height);
    tempCtx.drawImage(canvas, 0, 0, width, height);

    return tempCanvas.toDataURL('image/png');
  }, [width, height]);

  // Clear canvas
  const clearCanvas = useCallback(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);
    onClear();
  }, [width, height, onClear]);

  // Load image onto canvas
  const loadImage = useCallback(
    (dataURL: string) => {
      const ctx = ctxRef.current;
      if (!ctx) return;

      const img = new Image();
      img.onload = () => {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, width, height);

        const scale = Math.min(width / img.width, height / img.height);
        const x = (width - img.width * scale) / 2;
        const y = (height - img.height * scale) / 2;

        ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
        onImageLoad?.(dataURL);
      };
      img.src = dataURL;
    },
    [width, height, onImageLoad]
  );

  // Get cursor position relative to canvas
  const getPos = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };

      const rect = canvas.getBoundingClientRect();
      const scaleX = width / rect.width;
      const scaleY = height / rect.height;

      return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
      };
    },
    [width, height]
  );

  // Mouse down handler
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const pos = getPos(e);

    if (spaceKeyRef.current || tool === 'pan') {
      setIsPanning(true);
      panStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        offsetX: panOffset.x,
        offsetY: panOffset.y,
      };
      return;
    }

    const ctx = ctxRef.current;
    if (!ctx) return;

    ctx.lineWidth = brushSize;
    ctx.globalCompositeOperation = tool === 'eraser' ? 'destination-out' : 'source-over';
    ctx.strokeStyle = '#000000';

    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    ctx.lineTo(pos.x + 0.1, pos.y + 0.1);
    ctx.stroke();

    setIsDrawing(true);
  };

  // Mouse move handler
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getPos(e);
    setCursorPos(pos);

    if (isPanning) {
      const dx = e.clientX - panStartRef.current.x;
      const dy = e.clientY - panStartRef.current.y;
      setPanOffset({
        x: panStartRef.current.offsetX + dx,
        y: panStartRef.current.offsetY + dy,
      });
      return;
    }

    if (!isDrawing) return;

    const ctx = ctxRef.current;
    if (!ctx) return;

    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  };

  // Mouse up handler
  const handleMouseUp = () => {
    if (isDrawing) {
      const ctx = ctxRef.current;
      if (ctx) {
        ctx.closePath();
      }
      setIsDrawing(false);
    }
    if (isPanning) {
      setIsPanning(false);
    }
  };

  const handleMouseLeave = () => {
    setCursorPos(null);
    if (isDrawing) {
      const ctx = ctxRef.current;
      if (ctx) {
        ctx.closePath();
      }
      setIsDrawing(false);
    }
  };

  // Wheel zoom
  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    // Zoom handled by parent component
  };

  // Load image when loadImageURL changes
  useEffect(() => {
    if (loadImageURL) {
      loadImage(loadImageURL);
    }
  }, [loadImageURL, loadImage]);

  // Expose methods to parent
  useEffect(() => {
    onExport(exportToPNG());
  }, [exportToPNG, onExport]);

  const containerStyle: React.CSSProperties = {
    position: 'relative',
    width: width * zoom,
    height: height * zoom,
    cursor:
      spaceKeyRef.current || isPanning
        ? 'grab'
        : tool === 'eraser'
        ? 'cell'
        : tool === 'pan'
        ? 'grab'
        : 'crosshair',
  };

  return (
    <div style={containerStyle}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="absolute top-0 left-0 border border-slate-300 rounded-lg"
        style={{
          width: width * zoom,
          height: height * zoom,
          touchAction: 'none',
          background: '#ffffff',
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onWheel={handleWheel}
      />
      <Overlay
        width={width}
        height={height}
        zoom={zoom}
        cursorPos={cursorPos}
        tool={tool}
        brushSize={brushSize}
      />
    </div>
  );
}
