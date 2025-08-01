import React, { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { Table, Button, Modal, Input, Card, Statistic, Row, Col, Space, Tag, Tabs, Popconfirm } from 'antd';
import { 
  CheckOutlined, 
  EditOutlined, 
  DeleteOutlined, 
  ExclamationCircleOutlined,
  LikeOutlined,
  DislikeOutlined,
  LogoutOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { 
  useRawData, 
  useTrainingData, 
  useStats, 
  useUpdateAnswer,
  useDeleteRawData,
  useApproveToTraining,
  useDeleteTrainingData
} from './hooks/useApiQueries';
import { RawDataItem, TrainingDataItem } from './types/api';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import Login from './components/Login';
import DocumentViewer from './components/DocumentViewer';
import './App.css';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function AppContent() {
  const { isAuthenticated, logout } = useAuth();
  const [activeTab, setActiveTab] = useState<string>('rawdata');
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editAnswer, setEditAnswer] = useState('');
  
  // Pagination states
  const [rawDataPage, setRawDataPage] = useState(1);
  const [rawDataPageSize, setRawDataPageSize] = useState(20);
  const [trainingDataPage, setTrainingDataPage] = useState(1);
  const [trainingDataPageSize, setTrainingDataPageSize] = useState(20);
  
  // Query hooks
  const { data: rawDataResponse, isLoading: rawDataLoading } = useRawData(rawDataPage, rawDataPageSize, false);
  const { data: trainingDataResponse, isLoading: trainingDataLoading } = useTrainingData(trainingDataPage, trainingDataPageSize);
  const { data: stats } = useStats();
  
  // Mutation hooks
  const updateAnswerMutation = useUpdateAnswer();
  const deleteRawDataMutation = useDeleteRawData();
  const approveToTrainingMutation = useApproveToTraining();
  const deleteTrainingDataMutation = useDeleteTrainingData();
  
  if (!isAuthenticated) {
    return <Login />;
  }

  const handleEdit = (record: RawDataItem) => {
    setEditingId(record.id);
    setEditAnswer(record.answer);
  };

  const handleSaveEdit = async () => {
    if (editingId === null) return;
    
    await updateAnswerMutation.mutateAsync({ id: editingId, answer: editAnswer });
    setEditingId(null);
    setEditAnswer('');
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditAnswer('');
  };

  const handleApprove = async (record: RawDataItem) => {
    await approveToTrainingMutation.mutateAsync(record.id);
  };

  const handleDeleteRawData = async (id: number) => {
    await deleteRawDataMutation.mutateAsync(id);
  };

  const handleRemoveFromTraining = async (id: number) => {
    await deleteTrainingDataMutation.mutateAsync(id);
  };


  const rawDataColumns: ColumnsType<RawDataItem> = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 50,
      fixed: 'left',
    },
    {
      title: 'User',
      dataIndex: 'username',
      key: 'username',
      width: 100,
      ellipsis: true,
      render: (username: string) => username || <Tag color="default">Anon</Tag>,
    },
    {
      title: 'T.ID',
      dataIndex: 'telegram_id',
      key: 'telegram_id',
      width: 80,
      ellipsis: true,
    },
    {
      title: 'M.ID',
      dataIndex: 'telegram_message_id',
      key: 'telegram_message_id',
      width: 70,
      ellipsis: true,
    },
    {
      title: 'Question',
      dataIndex: 'question',
      key: 'question',
      width: 250,
      ellipsis: true,
      onCell: (record: RawDataItem) => ({
        onDoubleClick: () => {
          Modal.info({
            title: 'Full Question',
            content: record.question,
            width: 600,
          });
        },
      }),
    },
    {
      title: 'Answer',
      dataIndex: 'answer',
      key: 'answer',
      width: 300,
      ellipsis: true,
      render: (text: string, record: RawDataItem) => {
        if (editingId === record.id) {
          return (
            <Input.TextArea
              value={editAnswer}
              onChange={(e) => setEditAnswer(e.target.value)}
              autoSize={{ minRows: 2, maxRows: 6 }}
            />
          );
        }
        return <span>{text || <Tag color="warning">No answer</Tag>}</span>;
      },
      onCell: (record: RawDataItem) => ({
        onDoubleClick: () => {
          if (record.answer) {
            Modal.info({
              title: 'Full Answer',
              content: record.answer,
              width: 600,
            });
          }
        },
      }),
    },
    {
      title: 'Votes',
      key: 'votes',
      width: 80,
      render: (_: any, record: RawDataItem) => (
        <Space direction="vertical" size="small">
          <Space>
            <span style={{ color: '#52c41a' }}>👍 {record.likes || 0}</span>
            <span style={{ color: '#f5222d' }}>👎 {record.dislikes || 0}</span>
          </Space>
          <span style={{ fontWeight: 'bold' }}>
            Score: {record.vote_score || 0}
          </span>
        </Space>
      ),
    },
    {
      title: 'Status',
      key: 'status',
      width: 80,
      filters: [
        { text: 'Approved', value: 'approved' },
        { text: 'Pending', value: 'pending' },
      ],
      onFilter: (value: any, record: RawDataItem) => {
        if (value === 'approved') return record.admin_approved === 1;
        return record.admin_approved === 0;
      },
      render: (_: any, record: RawDataItem) => (
        record.admin_approved === 1 ? 
          <Tag color="success">OK</Tag> : 
          <Tag color="default">Pending</Tag>
      ),
    },
    {
      title: 'Duplicates',
      key: 'duplicates',
      width: 120,
      filters: [
        { text: 'Duplicate', value: 'duplicate' },
        { text: 'Original', value: 'original' },
      ],
      onFilter: (value: any, record: RawDataItem) => {
        if (value === 'duplicate') return record.is_duplicate === true;
        return record.is_duplicate === false;
      },
      render: (_: any, record: RawDataItem) => {
        if (record.is_duplicate) {
          return (
            <Space direction="vertical" size="small">
              <Tag color="warning">Duplicate</Tag>
              <Tag color="blue">→ ID: {record.duplicate_of_id}</Tag>
            </Space>
          );
        }
        return <Tag color="default">Original</Tag>;
      },
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 130,
      render: (text: string) => new Date(text).toLocaleString('tr-TR', { 
        day: '2-digit', 
        month: '2-digit', 
        year: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      }),
      sorter: (a: RawDataItem, b: RawDataItem) => 
        new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 180,
      fixed: 'right',
      render: (_: any, record: RawDataItem) => (
        <Space size="small">
          {editingId === record.id ? (
            <>
              <Button size="small" type="primary" onClick={handleSaveEdit}>
                Save
              </Button>
              <Button size="small" onClick={handleCancelEdit}>
                Cancel
              </Button>
            </>
          ) : (
            <>
              <Button
                size="small"
                icon={<EditOutlined />}
                onClick={() => handleEdit(record)}
                disabled={!record.answer}
              />
              <Popconfirm
                title="Approve to training data"
                description="Are you sure you want to approve this data for training?"
                onConfirm={() => handleApprove(record)}
                okText="Yes"
                cancelText="No"
              >
                <Button
                  size="small"
                  type="primary"
                  icon={<CheckOutlined />}
                  disabled={record.admin_approved === 1 || !record.answer}
                />
              </Popconfirm>
              <Popconfirm
                title="Delete raw data"
                description="Are you sure you want to delete this data?"
                onConfirm={() => handleDeleteRawData(record.id)}
                okText="Yes"
                cancelText="No"
              >
                <Button
                  size="small"
                  danger
                  icon={<DeleteOutlined />}
                />
              </Popconfirm>
            </>
          )}
        </Space>
      ),
    },
  ];

  const trainingDataColumns: ColumnsType<TrainingDataItem> = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 60,
    },
    {
      title: 'Question',
      dataIndex: 'question',
      key: 'question',
      width: 350,
      ellipsis: true,
      onCell: (record: TrainingDataItem) => ({
        onDoubleClick: () => {
          Modal.info({
            title: 'Full Question',
            content: record.question,
            width: 600,
          });
        },
      }),
    },
    {
      title: 'Answer',
      dataIndex: 'answer',
      key: 'answer',
      width: 450,
      ellipsis: true,
      onCell: (record: TrainingDataItem) => ({
        onDoubleClick: () => {
          Modal.info({
            title: 'Full Answer',
            content: record.answer,
            width: 600,
          });
        },
      }),
    },
    {
      title: 'Rating',
      dataIndex: 'point',
      key: 'point',
      width: 100,
      render: (point: number | null) => {
        if (point === 1) return <Tag color="success">Liked</Tag>;
        if (point === -1) return <Tag color="error">Disliked</Tag>;
        return <Tag>No rating</Tag>;
      },
    },
    {
      title: 'Duplicates',
      key: 'duplicates',
      width: 150,
      filters: [
        { text: 'Duplicate', value: 'duplicate' },
        { text: 'Original', value: 'original' },
      ],
      onFilter: (value: any, record: TrainingDataItem) => {
        if (value === 'duplicate') return record.is_answer_duplicate === true;
        return record.is_answer_duplicate === false;
      },
      render: (_: any, record: TrainingDataItem) => {
        if (record.is_answer_duplicate) {
          return (
            <Space direction="vertical" size="small">
              <Tag color="warning">Duplicate</Tag>
              <Tag color="blue">→ ID: {record.duplicate_answer_of_id}</Tag>
            </Space>
          );
        }
        return <Tag color="default">Original</Tag>;
      },
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 150,
      render: (text: string) => new Date(text).toLocaleString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (_: any, record: TrainingDataItem) => (
        <Popconfirm
          title="Remove from training data"
          description="Are you sure you want to remove this from training data?"
          onConfirm={() => handleRemoveFromTraining(record.id)}
          okText="Yes"
          cancelText="No"
        >
          <Button size="small" danger icon={<DeleteOutlined />}>
            Remove
          </Button>
        </Popconfirm>
      ),
    },
  ];

  return (
    <div className="App">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <h1>University Bot Admin Panel</h1>
        <Button 
          type="primary" 
          danger 
          icon={<LogoutOutlined />}
          onClick={logout}
        >
          Logout
        </Button>
      </div>
      
      {stats && (
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={4}>
            <Card>
              <Statistic title="Total Questions" value={stats.total_questions} />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic 
                title="Approved" 
                value={stats.approved_questions} 
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic 
                title="Liked" 
                value={stats.liked_questions} 
                prefix={<LikeOutlined />}
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic 
                title="Disliked" 
                value={stats.disliked_questions} 
                prefix={<DislikeOutlined />}
                valueStyle={{ color: '#cf1322' }}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic title="Training Data" value={stats.training_data_count} />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic 
                title="Duplicates" 
                value={stats.duplicate_count} 
                valueStyle={{ color: '#faad14' }}
              />
            </Card>
          </Col>
        </Row>
      )}

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'rawdata',
            label: 'Raw Data',
            children: (
              <Table
                columns={rawDataColumns}
                dataSource={rawDataResponse?.data || []}
                rowKey="id"
                loading={rawDataLoading}
                scroll={{ x: 1400 }}
                pagination={{
                  current: rawDataPage,
                  pageSize: rawDataPageSize,
                  total: rawDataResponse?.total || 0,
                  showSizeChanger: true,
                  showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} items`,
                  pageSizeOptions: ['10', '20', '50', '100'],
                  onChange: (page, pageSize) => {
                    setRawDataPage(page);
                    setRawDataPageSize(pageSize);
                  },
                }}
                rowClassName={(record) => {
                  if (record.admin_approved === 1) return 'approved-row';
                  if (!record.answer) return 'unanswered-row';
                  return '';
                }}
              />
            ),
          },
          {
            key: 'training',
            label: 'Training Data',
            children: (
              <Table
                columns={trainingDataColumns}
                dataSource={trainingDataResponse?.data || []}
                rowKey="id"
                loading={trainingDataLoading}
                scroll={{ x: 1400 }}
                pagination={{
                  current: trainingDataPage,
                  pageSize: trainingDataPageSize,
                  total: trainingDataResponse?.total || 0,
                  showSizeChanger: true,
                  showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} items`,
                  pageSizeOptions: ['10', '20', '50', '100'],
                  onChange: (page, pageSize) => {
                    setTrainingDataPage(page);
                    setTrainingDataPageSize(pageSize);
                  },
                }}
              />
            ),
          },
          {
            key: 'docs',
            label: 'Documentation',
            children: <DocumentViewer />,
          },
        ]}
      />
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <AppContent />
        <ReactQueryDevtools initialIsOpen={false} />
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;