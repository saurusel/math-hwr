interface OverlayProps {
  width: number;
  height: number;
  zoom: number;
  cursorPos: { x: number; y: number } | null;
  tool: 'brush' | 'eraser' | 'pan';
  brushSize: number;
}

export function Overlay({ width, height, zoom, cursorPos, tool, brushSize }: OverlayProps) {
  if (!cursorPos || tool !== 'eraser') return null;

  const circleStyle: React.CSSProperties = {
    position: 'absolute',
    left: cursorPos.x * zoom - (brushSize * zoom) / 2,
    top: cursorPos.y * zoom - (brushSize * zoom) / 2,
    width: brushSize * zoom,
    height: brushSize * zoom,
    border: '1px solid rgba(30, 41, 59, 0.9)',
    borderRadius: '50%',
    pointerEvents: 'none',
    zIndex: 10,
  };

  return <div style={circleStyle} />;
}
