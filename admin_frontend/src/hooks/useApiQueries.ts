import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiService } from '../services/api';
import { message } from 'antd';

// Query keys
export const queryKeys = {
  rawData: (page: number, pageSize: number, onlyUnapproved: boolean) => 
    ['rawData', page, pageSize, onlyUnapproved] as const,
  trainingData: (page: number, pageSize: number) => 
    ['trainingData', page, pageSize] as const,
  stats: ['stats'] as const,
};

// Raw Data Queries
export const useRawData = (page: number, pageSize: number, onlyUnapproved: boolean) => {
  return useQuery({
    queryKey: queryKeys.rawData(page, pageSize, onlyUnapproved),
    queryFn: () => apiService.getRawData(page, pageSize, onlyUnapproved),
    staleTime: 30000, // Consider data stale after 30 seconds
    refetchInterval: 30000, // Auto-refresh every 30 seconds
  });
};

export const useUpdateAnswer = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ id, answer }: { id: number; answer: string }) => 
      apiService.updateAnswer(id, { answer }),
    onSuccess: () => {
      message.success('Answer updated successfully');
      // Invalidate raw data queries to refetch
      queryClient.invalidateQueries({ queryKey: ['rawData'] });
    },
    onError: () => {
      message.error('Failed to update answer');
    },
  });
};

export const useDeleteRawData = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: number) => apiService.deleteRawData(id),
    onSuccess: () => {
      message.success('Raw data deleted successfully');
      queryClient.invalidateQueries({ queryKey: ['rawData'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
    },
    onError: () => {
      message.error('Failed to delete raw data');
    },
  });
};

export const useApproveToTraining = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: number) => apiService.approveToTraining(id),
    onSuccess: () => {
      message.success('Approved to training data');
      queryClient.invalidateQueries({ queryKey: ['rawData'] });
      queryClient.invalidateQueries({ queryKey: ['trainingData'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
    },
    onError: () => {
      message.error('Failed to approve');
    },
  });
};

// Training Data Queries
export const useTrainingData = (page: number, pageSize: number) => {
  return useQuery({
    queryKey: queryKeys.trainingData(page, pageSize),
    queryFn: () => apiService.getTrainingData(page, pageSize),
    staleTime: 30000,
    refetchInterval: 30000,
  });
};

export const useDeleteTrainingData = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (id: number) => apiService.deleteTrainingData(id),
    onSuccess: () => {
      message.success('Training data removed successfully');
      queryClient.invalidateQueries({ queryKey: ['trainingData'] });
      queryClient.invalidateQueries({ queryKey: ['rawData'] });
      queryClient.invalidateQueries({ queryKey: ['stats'] });
    },
    onError: () => {
      message.error('Failed to remove training data');
    },
  });
};

// Stats Query
export const useStats = () => {
  return useQuery({
    queryKey: queryKeys.stats,
    queryFn: apiService.getStats,
    staleTime: 30000,
    refetchInterval: 30000,
  });
};

