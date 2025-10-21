import { useRef, useCallback } from 'react';

export type Tool = 'brush' | 'eraser' | 'pan';

export interface CanvasState {
  zoom: number;
  panX: number;
  panY: number;
  brushSize: number;
  tool: Tool;
}

export function useCanvasTools(initialState: Partial<CanvasState> = {}) {
  const stateRef = useRef<CanvasState>({
    zoom: initialState.zoom ?? 1,
    panX: initialState.panX ?? 0,
    panY: initialState.panY ?? 0,
    brushSize: initialState.brushSize ?? 3,
    tool: initialState.tool ?? 'brush',
  });

  const getState = useCallback(() => stateRef.current, []);

  const setZoom = useCallback((zoom: number) => {
    stateRef.current.zoom = Math.max(0.25, Math.min(4, zoom));
  }, []);

  const setPan = useCallback((panX: number, panY: number) => {
    stateRef.current.panX = panX;
    stateRef.current.panY = panY;
  }, []);

  const setBrushSize = useCallback((size: number) => {
    stateRef.current.brushSize = Math.max(1, Math.min(50, size));
  }, []);

  const setTool = useCallback((tool: Tool) => {
    stateRef.current.tool = tool;
  }, []);

  const resetView = useCallback(() => {
    stateRef.current.zoom = 1;
    stateRef.current.panX = 0;
    stateRef.current.panY = 0;
  }, []);

  return {
    getState,
    setZoom,
    setPan,
    setBrushSize,
    setTool,
    resetView,
  };
}
