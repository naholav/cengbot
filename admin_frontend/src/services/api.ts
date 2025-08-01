import axios from 'axios';
import { 
  RawDataItem, 
  TrainingDataItem, 
  Stats, 
  UpdateAnswerRequest, 
  ApproveResponse,
  PaginatedResponse 
} from '../types/api';

const API_BASE_URL = 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth header to all requests
api.interceptors.request.use((config) => {
  const authHeader = localStorage.getItem('adminAuth');
  if (authHeader) {
    config.headers.Authorization = authHeader;
  }
  return config;
});

export const apiService = {
  // Raw Data
  getRawData: async (page: number = 1, pageSize: number = 20, onlyUnapproved: boolean = false) => {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString(),
      only_unapproved: onlyUnapproved.toString(),
    });
    const response = await api.get<PaginatedResponse<RawDataItem>>(`/raw-data?${params}`);
    return response.data;
  },

  updateAnswer: async (id: number, data: UpdateAnswerRequest) => {
    const response = await api.put(`/raw-data/${id}`, data);
    return response.data;
  },

  deleteRawData: async (id: number) => {
    const response = await api.delete(`/raw-data/${id}`);
    return response.data;
  },

  approveToTraining: async (id: number) => {
    const response = await api.post<ApproveResponse>(`/approve/${id}`);
    return response.data;
  },

  // Training Data
  getTrainingData: async (page: number = 1, pageSize: number = 20) => {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString(),
    });
    const response = await api.get<PaginatedResponse<TrainingDataItem>>(`/training-data?${params}`);
    return response.data;
  },

  deleteTrainingData: async (id: number) => {
    const response = await api.delete(`/training-data/${id}`);
    return response.data;
  },

  // Stats
  getStats: async () => {
    const response = await api.get<Stats>('/stats');
    return response.data;
  },

};