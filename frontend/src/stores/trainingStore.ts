import { create } from 'zustand';
import type {
  TrainingJob,
  MetricEvent,
  LogEvent,
  SamplePredEvent,
  CheckpointEvent,
} from '../types/training';

interface TrainingState {
  currentJob: TrainingJob | null;
  metrics: MetricEvent[];
  logs: LogEvent[];
  samples: SamplePredEvent[];
  checkpoints: CheckpointEvent[];

  setCurrentJob: (job: TrainingJob | null) => void;
  addMetric: (metric: MetricEvent) => void;
  addLog: (log: LogEvent) => void;
  addSamples: (samples: SamplePredEvent) => void;
  addCheckpoint: (checkpoint: CheckpointEvent) => void;
  clearJobData: () => void;
}

export const useTrainingStore = create<TrainingState>((set) => ({
  currentJob: null,
  metrics: [],
  logs: [],
  samples: [],
  checkpoints: [],

  setCurrentJob: (job) => set({ currentJob: job }),

  addMetric: (metric) =>
    set((state) => {
      console.log('[Store] Adding metric:', metric); // Debug logging

      // Check for duplicates - don't add if epoch already exists
      const isDuplicate = state.metrics.some(
        (m) => m.epoch === metric.epoch
      );

      if (isDuplicate) {
        console.log('[Store] Skipping duplicate metric for epoch:', metric.epoch);
        return state; // Don't update if duplicate
      }

      return {
        metrics: [...state.metrics, metric],
        // Update currentJob epoch from metric
        currentJob: state.currentJob
          ? { ...state.currentJob, current_epoch: metric.epoch }
          : null,
      };
    }),

  addLog: (log) =>
    set((state) => {
      console.log('[Store] Adding log:', log.line); // Debug logging
      return {
        logs: [...state.logs.slice(-500), log], // Keep last 500 logs
      };
    }),

  addSamples: (samples) =>
    set((state) => {
      console.log('[Store] Adding samples:', samples.items.length, 'items'); // Debug logging
      return {
        samples: [...state.samples.slice(-10), samples], // Keep last 10 sample batches
      };
    }),

  addCheckpoint: (checkpoint) =>
    set((state) => ({
      checkpoints: [...state.checkpoints, checkpoint],
    })),

  clearJobData: () =>
    set({
      metrics: [],
      logs: [],
      samples: [],
      checkpoints: [],
    }),
}));
