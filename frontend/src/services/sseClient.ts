import type { MetricEvent, LogEvent, SamplePredEvent, CheckpointEvent, StatusEvent } from '../types/training';

const BASE_URL = 'http://127.0.0.1:8000';

export type SSEEventType = 'metric' | 'log' | 'sample_pred' | 'checkpoint' | 'status';

export interface SSEEventHandlers {
  onMetric?: (data: MetricEvent) => void;
  onLog?: (data: LogEvent) => void;
  onSamplePred?: (data: SamplePredEvent) => void;
  onCheckpoint?: (data: CheckpointEvent) => void;
  onStatus?: (data: StatusEvent) => void;
  onError?: (error: Error) => void;
  onOpen?: () => void;
}

export class TrainingSSEClient {
  private eventSource: EventSource | null = null;
  private jobId: string;
  private handlers: SSEEventHandlers;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor(jobId: string, handlers: SSEEventHandlers) {
    this.jobId = jobId;
    this.handlers = handlers;
  }

  connect() {
    try {
      const url = `${BASE_URL}/api/train/stream/${this.jobId}`;
      this.eventSource = new EventSource(url);

      this.eventSource.addEventListener('metric', (e) => {
        try {
          const data: MetricEvent = JSON.parse(e.data);
          console.log('[SSE] Metric received:', data); // Debug logging
          this.handlers.onMetric?.(data);
        } catch (error) {
          console.error('Failed to parse metric event:', error);
        }
      });

      this.eventSource.addEventListener('log', (e) => {
        try {
          const data: LogEvent = JSON.parse(e.data);
          this.handlers.onLog?.(data);
        } catch (error) {
          console.error('Failed to parse log event:', error);
        }
      });

      this.eventSource.addEventListener('sample_pred', (e) => {
        try {
          const data: SamplePredEvent = JSON.parse(e.data);
          console.log('[SSE] Sample predictions received:', data.items.length, 'samples'); // Debug logging
          this.handlers.onSamplePred?.(data);
        } catch (error) {
          console.error('Failed to parse sample_pred event:', error);
        }
      });

      this.eventSource.addEventListener('checkpoint', (e) => {
        try {
          const data: CheckpointEvent = JSON.parse(e.data);
          this.handlers.onCheckpoint?.(data);
        } catch (error) {
          console.error('Failed to parse checkpoint event:', error);
        }
      });

      this.eventSource.addEventListener('status', (e) => {
        try {
          const data: StatusEvent = JSON.parse(e.data);
          this.handlers.onStatus?.(data);
        } catch (error) {
          console.error('Failed to parse status event:', error);
        }
      });

      this.eventSource.onopen = () => {
        this.reconnectAttempts = 0;
        console.log('[SSE] Connection established'); // Debug logging
        this.handlers.onOpen?.();
      };

      this.eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        this.handleReconnect();
      };
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Unknown SSE error');
      this.handlers.onError?.(err);
    }
  }

  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Reconnecting SSE (attempt ${this.reconnectAttempts})...`);
        this.disconnect();
        this.connect();
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      this.handlers.onError?.(new Error('Max reconnection attempts reached'));
    }
  }

  disconnect() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }

  isConnected(): boolean {
    return this.eventSource !== null && this.eventSource.readyState === EventSource.OPEN;
  }
}
