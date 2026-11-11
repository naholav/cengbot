import { useQuery, useMutation, useQueryClient, keepPreviousData } from '@tanstack/react-query';
import { message } from 'antd';
import { apiService, getErrorMessage } from '../services/api';

/** Auto-refresh period for the data tables and stats (only while visible). */
export const POLL_INTERVAL_MS = 30000;
const STALE_TIME_MS = 30000;

// Query keys (all parameters are part of the key so pages/filters are cached separately)
export const queryKeys = {
  rawData: (page: number, pageSize: number, onlyUnapproved: boolean) =>
    ['rawData', page, pageSize, onlyUnapproved] as const,
  trainingData: (page: number, pageSize: number) =>
    ['trainingData', page, pageSize] as const,
  stats: ['stats'] as const,
  documents: ['documents'] as const,
  documentContent: (path: string) => ['documents', 'content', path] as const,
};

export interface PollingOptions {
  /** Set to false to stop auto-refresh (hidden tab, row being edited, ...). */
  poll?: boolean;
}

// Raw Data Queries
export const useRawData = (
  page: number,
  pageSize: number,
  onlyUnapproved: boolean,
  { poll = true }: PollingOptions = {}
) => {
  return useQuery({
    queryKey: queryKeys.rawData(page, pageSize, onlyUnapproved),
    queryFn: () => apiService.getRawData(page, pageSize, onlyUnapproved),
    staleTime: STALE_TIME_MS,
    refetchInterval: poll ? POLL_INTERVAL_MS : false,
    // Keep showing the current page while the next one loads (no empty flash).
    placeholderData: keepPreviousData,
  });
};

export const useUpdateAnswer = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, answer }: { id: number; answer: string }) =>
      apiService.updateAnswer(id, { answer }),
    onSuccess: (data) => {
      message.success('Answer updated successfully');
      const refreshes = [queryClient.invalidateQueries({ queryKey: ['rawData'] })];
      if (data?.training_data_updated) {
        // The backend also rewrote the approved training copy.
        refreshes.push(queryClient.invalidateQueries({ queryKey: ['trainingData'] }));
      }
      return Promise.all(refreshes);
    },
    onError: (error) => {
      message.error(getErrorMessage(error, 'Failed to update answer'));
    },
  });
};

export const useDeleteRawData = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => apiService.deleteRawData(id),
    onSuccess: () => {
      message.success('Raw data deleted successfully');
      // Deleting raw data also deletes its training copy on the backend.
      return Promise.all([
        queryClient.invalidateQueries({ queryKey: ['rawData'] }),
        queryClient.invalidateQueries({ queryKey: ['trainingData'] }),
        queryClient.invalidateQueries({ queryKey: ['stats'] }),
      ]);
    },
    onError: (error) => {
      message.error(getErrorMessage(error, 'Failed to delete raw data'));
    },
  });
};

export const useApproveToTraining = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => apiService.approveToTraining(id),
    onSuccess: () => {
      message.success('Approved to training data');
      return Promise.all([
        queryClient.invalidateQueries({ queryKey: ['rawData'] }),
        queryClient.invalidateQueries({ queryKey: ['trainingData'] }),
        queryClient.invalidateQueries({ queryKey: ['stats'] }),
      ]);
    },
    onError: (error) => {
      message.error(getErrorMessage(error, 'Failed to approve'));
    },
  });
};

// Training Data Queries
export const useTrainingData = (
  page: number,
  pageSize: number,
  { poll = true }: PollingOptions = {}
) => {
  return useQuery({
    queryKey: queryKeys.trainingData(page, pageSize),
    queryFn: () => apiService.getTrainingData(page, pageSize),
    staleTime: STALE_TIME_MS,
    refetchInterval: poll ? POLL_INTERVAL_MS : false,
    placeholderData: keepPreviousData,
  });
};

export const useDeleteTrainingData = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => apiService.deleteTrainingData(id),
    onSuccess: () => {
      message.success('Training data removed successfully');
      return Promise.all([
        queryClient.invalidateQueries({ queryKey: ['trainingData'] }),
        queryClient.invalidateQueries({ queryKey: ['rawData'] }),
        queryClient.invalidateQueries({ queryKey: ['stats'] }),
      ]);
    },
    onError: (error) => {
      message.error(getErrorMessage(error, 'Failed to remove training data'));
    },
  });
};

// Stats Query
export const useStats = ({ poll = true }: PollingOptions = {}) => {
  return useQuery({
    queryKey: queryKeys.stats,
    queryFn: apiService.getStats,
    staleTime: STALE_TIME_MS,
    refetchInterval: poll ? POLL_INTERVAL_MS : false,
  });
};

// Documentation Queries
export const useDocuments = () => {
  return useQuery({
    queryKey: queryKeys.documents,
    queryFn: apiService.listDocuments,
    staleTime: 60000,
  });
};

export const useDocumentContent = (path: string | null) => {
  return useQuery({
    queryKey: queryKeys.documentContent(path ?? ''),
    queryFn: () => apiService.readDocument(path as string),
    enabled: path !== null,
    staleTime: 60000,
    // Document bodies can be large; don't keep them cached after the modal closes.
    gcTime: 0,
  });
};
