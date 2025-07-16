import React, { useState, useEffect } from 'react';
import { Table, Button, Modal, Input, message, Card, Statistic, Row, Col, Space, Tag, Tabs } from 'antd';
import { CheckOutlined, EditOutlined, LikeOutlined, DislikeOutlined, DeleteOutlined, ExclamationCircleOutlined } from '@ant-design/icons';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8001';

interface RawData {
  id: number;
  telegram_id: number;
  username: string;
  question: string;
  answer: string;
  like: number | null;
  admin_approved: number;
  is_duplicate: boolean;
  duplicate_of_id: number | null;
  created_at: string;
  answered_at: string;
}

interface TrainingData {
  id: number;
  question: string;
  answer: string;
  created_at: string;
}

interface Stats {
  total_questions: number;
  approved_questions: number;
  liked_questions: number;
  disliked_questions: number;
  training_data_count: number;
  duplicate_count: number;
  approval_rate: string;
}

function App() {
  const [data, setData] = useState<RawData[]>([]);
  const [trainingData, setTrainingData] = useState<TrainingData[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(false);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editAnswer, setEditAnswer] = useState('');
  const [activeTab, setActiveTab] = useState('rawdata');

  useEffect(() => {
    fetchData();
    fetchStats();
    fetchTrainingData();
    // Auto refresh every 30 seconds
    const interval = setInterval(() => {
      fetchData();
      fetchStats();
      fetchTrainingData();
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_URL}/raw-data`);
      setData(response.data);
    } catch (error) {
      message.error('Failed to load data!');
      console.error(error);
    }
    setLoading(false);
  };

  const fetchTrainingData = async () => {
    setTrainingLoading(true);
    try {
      const response = await axios.get(`${API_URL}/training-data`);
      setTrainingData(response.data);
    } catch (error) {
      message.error('Failed to load training data!');
      console.error(error);
    }
    setTrainingLoading(false);
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_URL}/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const handleEdit = (record: RawData) => {
    setEditingId(record.id);
    setEditAnswer(record.answer || '');
  };

  const handleSaveEdit = async () => {
    if (!editingId) return;
    
    try {
      await axios.put(`${API_URL}/raw-data/${editingId}`, {
        answer: editAnswer
      });
      message.success('Answer updated successfully!');
      setEditingId(null);
      fetchData();
    } catch (error) {
      message.error('Update failed!');
      console.error(error);
    }
  };

  const handleApprove = async (id: number) => {
    Modal.confirm({
      title: 'Approve Question',
      content: 'Are you sure about the quality of this question-answer pair? It will be added to training data.',
      okText: 'Yes, I\'m sure',
      cancelText: 'Cancel',
      onOk: async () => {
        try {
          await axios.post(`${API_URL}/approve/${id}`);
          message.success('Added to training data!');
          fetchData();
          fetchStats();
          fetchTrainingData();
        } catch (error: any) {
          message.error(error.response?.data?.detail || 'Approval failed!');
        }
      }
    });
  };

  const handleDeleteTrainingData = async (id: number) => {
    Modal.confirm({
      title: 'Delete Training Data',
      content: 'Are you sure you want to delete this training data? This action cannot be undone.',
      okText: 'Yes, Delete',
      cancelText: 'Cancel',
      icon: <ExclamationCircleOutlined />,
      okType: 'danger',
      onOk: async () => {
        try {
          await axios.delete(`${API_URL}/training-data/${id}`);
          message.success('Training data deleted successfully!');
          fetchTrainingData();
          fetchStats();
        } catch (error: any) {
          message.error(error.response?.data?.detail || 'Delete failed!');
        }
      }
    });
  };

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 60,
      sorter: (a: RawData, b: RawData) => a.id - b.id,
    },
    {
      title: 'User',
      dataIndex: 'username',
      key: 'username',
      width: 120,
      render: (text: string) => text || 'Anonymous',
    },
    {
      title: 'Question',
      dataIndex: 'question',
      key: 'question',
      width: 300,
      ellipsis: true,
    },
    {
      title: 'Answer',
      dataIndex: 'answer',
      key: 'answer',
      width: 400,
      ellipsis: true,
      render: (text: string, record: RawData) => {
        if (editingId === record.id) {
          return (
            <Space direction="vertical" style={{ width: '100%' }}>
              <Input.TextArea
                value={editAnswer}
                onChange={(e) => setEditAnswer(e.target.value)}
                rows={4}
                placeholder="Edit the answer..."
              />
              <Space>
                <Button size="small" type="primary" onClick={handleSaveEdit}>Save</Button>
                <Button size="small" onClick={() => setEditingId(null)}>Cancel</Button>
              </Space>
            </Space>
          );
        }
        return text || <span style={{ color: '#999' }}>Waiting for answer...</span>;
      }
    },
    {
      title: 'Rating',
      dataIndex: 'like',
      key: 'like',
      width: 80,
      align: 'center' as const,
      render: (like: number | null) => {
        if (like === 1) return <LikeOutlined style={{ color: '#52c41a', fontSize: 18 }} />;
        if (like === -1) return <DislikeOutlined style={{ color: '#f5222d', fontSize: 18 }} />;
        return '-';
      },
      sorter: (a: RawData, b: RawData) => (a.like || -1) - (b.like || -1),
    },
    {
      title: 'Status',
      key: 'status',
      width: 150,
      render: (record: RawData) => (
        <Space>
          {record.admin_approved === 1 && <Tag color="green">Approved</Tag>}
          {record.is_duplicate && <Tag color="orange">Duplicate</Tag>}
          {!record.answer && <Tag color="red">Unanswered</Tag>}
        </Space>
      ),
      filters: [
        { text: 'Approved', value: 'approved' },
        { text: 'Pending Approval', value: 'pending' },
        { text: 'Duplicate', value: 'duplicate' },
      ],
      onFilter: (value: any, record: RawData) => {
        if (value === 'approved') return record.admin_approved === 1;
        if (value === 'pending') return record.admin_approved === 0;
        if (value === 'duplicate') return record.is_duplicate;
        return false;
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 200,
      fixed: 'right' as const,
      render: (record: RawData) => (
        <Space>
          <Button
            icon={<EditOutlined />}
            size="small"
            onClick={() => handleEdit(record)}
            disabled={!record.answer}
          >
            Edit
          </Button>
          <Button
            icon={<CheckOutlined />}
            size="small"
            type="primary"
            onClick={() => handleApprove(record.id)}
            disabled={record.admin_approved === 1 || !record.answer}
          >
            Approve
          </Button>
        </Space>
      )
    },
    {
      title: 'Date',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 150,
      render: (date: string) => new Date(date).toLocaleString('en-US'),
      sorter: (a: RawData, b: RawData) => 
        new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      defaultSortOrder: 'descend' as const,
    }
  ];

  const trainingColumns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 60,
      sorter: (a: TrainingData, b: TrainingData) => a.id - b.id,
    },
    {
      title: 'Question',
      dataIndex: 'question',
      key: 'question',
      width: 400,
      ellipsis: true,
    },
    {
      title: 'Answer',
      dataIndex: 'answer',
      key: 'answer',
      width: 400,
      ellipsis: true,
    },
    {
      title: 'Date',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 150,
      render: (date: string) => new Date(date).toLocaleString('en-US'),
      sorter: (a: TrainingData, b: TrainingData) => 
        new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      defaultSortOrder: 'descend' as const,
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 100,
      fixed: 'right' as const,
      render: (record: TrainingData) => (
        <Button
          icon={<DeleteOutlined />}
          size="small"
          danger
          onClick={() => handleDeleteTrainingData(record.id)}
        >
          Delete
        </Button>
      )
    }
  ];

  return (
    <div className="App">
      <div className="header">
        <h1>🤖 CengBot Admin Panel</h1>
        <p>Çukurova University Computer Engineering Bot Management System</p>
      </div>
      
      {stats && (
        <div className="stats-container">
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={12} md={8} lg={4}>
              <Card className="stat-card">
                <Statistic 
                  title="Total Questions" 
                  value={stats.total_questions} 
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={4}>
              <Card className="stat-card">
                <Statistic 
                  title="Approved" 
                  value={stats.approved_questions} 
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={4}>
              <Card className="stat-card">
                <Statistic 
                  title="Liked" 
                  value={stats.liked_questions} 
                  prefix={<LikeOutlined />}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={4}>
              <Card className="stat-card">
                <Statistic 
                  title="Disliked" 
                  value={stats.disliked_questions} 
                  prefix={<DislikeOutlined />}
                  valueStyle={{ color: '#f5222d' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={4}>
              <Card className="stat-card">
                <Statistic 
                  title="Training Data" 
                  value={stats.training_data_count} 
                  valueStyle={{ color: '#722ed1' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={4}>
              <Card className="stat-card">
                <Statistic 
                  title="Approval Rate" 
                  value={stats.approval_rate} 
                  valueStyle={{ color: '#fa8c16' }}
                />
              </Card>
            </Col>
          </Row>
        </div>
      )}
      
      <div className="table-container">
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={[
            {
              key: 'rawdata',
              label: '📋 Question-Answer List',
              children: (
                <div>
                  <div className="table-header">
                    <h2>📋 Question-Answer List</h2>
                    <Button type="primary" onClick={fetchData} loading={loading}>
                      Refresh
                    </Button>
                  </div>
                  
                  <Table
                    columns={columns}
                    dataSource={data}
                    rowKey="id"
                    loading={loading}
                    scroll={{ x: 1500 }}
                    pagination={{
                      pageSize: 20,
                      showSizeChanger: true,
                      showTotal: (total) => `Total ${total} records`,
                      pageSizeOptions: ['10', '20', '50', '100']
                    }}
                    rowClassName={(record) => {
                      if (record.admin_approved === 1) return 'approved-row';
                      if (!record.answer) return 'no-answer-row';
                      return '';
                    }}
                  />
                </div>
              )
            },
            {
              key: 'training',
              label: '🎓 Training Data',
              children: (
                <div>
                  <div className="table-header">
                    <h2>🎓 Training Data</h2>
                    <Button type="primary" onClick={fetchTrainingData} loading={trainingLoading}>
                      Refresh
                    </Button>
                  </div>
                  
                  <Table
                    columns={trainingColumns}
                    dataSource={trainingData}
                    rowKey="id"
                    loading={trainingLoading}
                    scroll={{ x: 1000 }}
                    pagination={{
                      pageSize: 20,
                      showSizeChanger: true,
                      showTotal: (total) => `Total ${total} training records`,
                      pageSizeOptions: ['10', '20', '50', '100']
                    }}
                  />
                </div>
              )
            }
          ]}
        />
      </div>
    </div>
  );
}

export default App;