import React, { lazy, useEffect, useState } from 'react';
import { Table, Button, Modal, Input, Card, Statistic, Row, Col, Space, Tag, Tabs, Popconfirm, Alert } from 'antd';
import {
  CheckOutlined,
  EditOutlined,
  DeleteOutlined,
  LikeOutlined,
  DislikeOutlined,
  LogoutOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type { FilterValue } from 'antd/es/table/interface';
import {
  useRawData,
  useTrainingData,
  useStats,
  useUpdateAnswer,
  useDeleteRawData,
  useApproveToTraining,
  useDeleteTrainingData
} from '../hooks/useApiQueries';
import { MAX_ANSWER_LENGTH, RawDataItem, TrainingDataItem } from '../types/api';
import { useAuth } from '../contexts/AuthContext';
import { getErrorMessage } from '../services/api';
import { formatDateTime, serverDateToMillis } from '../utils/format';
import LazyScreen from './LazyScreen';

// Loaded when the Documentation tab is first opened (antd renders a tab's
// content only once it becomes active).
const DocumentViewer = lazy(() => import('./DocumentViewer'));

/** Maximum page for a server-side paginated table (at least 1). */
export const clampPage = (page: number, totalPages: number | undefined): number => {
  if (totalPages === undefined) return page;
  return Math.min(page, Math.max(1, totalPages));
};

const QueryErrorAlert: React.FC<{ title: string; error: unknown }> = ({ title, error }) => (
  <Alert
    type="error"
    showIcon
    message={title}
    description={getErrorMessage(error)}
    style={{ marginBottom: 16 }}
  />
);

/** The logged-in screen (loaded lazily by App, see App.tsx). */
function Dashboard() {
  const { logout } = useAuth();
  const [activeTab, setActiveTab] = useState<string>('rawdata');
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editAnswer, setEditAnswer] = useState('');

  // Pagination / filter states
  const [rawDataPage, setRawDataPage] = useState(1);
  const [rawDataPageSize, setRawDataPageSize] = useState(20);
  const [onlyUnapproved, setOnlyUnapproved] = useState(false);
  const [rawDuplicateFilter, setRawDuplicateFilter] = useState<FilterValue | null>(null);
  const [trainingDataPage, setTrainingDataPage] = useState(1);
  const [trainingDataPageSize, setTrainingDataPageSize] = useState(20);

  // Query hooks. Polling is paused while an answer is being edited (new
  // questions would otherwise shift the edited row to another page) and for
  // tables on hidden tabs.
  const rawDataQuery = useRawData(rawDataPage, rawDataPageSize, onlyUnapproved, {
    poll: activeTab === 'rawdata' && editingId === null,
  });
  const trainingDataQuery = useTrainingData(trainingDataPage, trainingDataPageSize, {
    poll: activeTab === 'training',
  });
  const statsQuery = useStats({ poll: editingId === null });
  const rawDataResponse = rawDataQuery.data;
  const trainingDataResponse = trainingDataQuery.data;
  const stats = statsQuery.data;

  // Mutation hooks
  const updateAnswerMutation = useUpdateAnswer();
  const deleteRawDataMutation = useDeleteRawData();
  const approveToTrainingMutation = useApproveToTraining();
  const deleteTrainingDataMutation = useDeleteTrainingData();

  // After deletions the current page can lie beyond the last page: step back.
  const rawTotalPages = rawDataResponse?.total_pages;
  useEffect(() => {
    const clamped = clampPage(rawDataPage, rawTotalPages);
    if (clamped !== rawDataPage) setRawDataPage(clamped);
  }, [rawDataPage, rawTotalPages]);

  const trainingTotalPages = trainingDataResponse?.total_pages;
  useEffect(() => {
    const clamped = clampPage(trainingDataPage, trainingTotalPages);
    if (clamped !== trainingDataPage) setTrainingDataPage(clamped);
  }, [trainingDataPage, trainingTotalPages]);

  const handleEdit = (record: RawDataItem) => {
    setEditingId(record.id);
    setEditAnswer(record.answer ?? '');
  };

  const handleSaveEdit = async () => {
    if (editingId === null || !editAnswer.trim()) return;
    try {
      await updateAnswerMutation.mutateAsync({ id: editingId, answer: editAnswer });
      setEditingId(null);
      setEditAnswer('');
    } catch {
      // Error toast is shown by the mutation; keep the editor open.
    }
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditAnswer('');
  };

  // Returning the promise lets Popconfirm show a spinner until it settles;
  // errors are reported by the mutation's onError toast.
  const handleApprove = (record: RawDataItem) =>
    approveToTrainingMutation.mutateAsync(record.id).catch(() => undefined);

  const handleDeleteRawData = (id: number) =>
    deleteRawDataMutation.mutateAsync(id)
      .then(() => {
        if (editingId === id) handleCancelEdit();
      })
      .catch(() => undefined);

  const handleRemoveFromTraining = (id: number) =>
    deleteTrainingDataMutation.mutateAsync(id).catch(() => undefined);

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
      render: (username: string | null) => username || <Tag color="default">Anon</Tag>,
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
      render: (text: string | null, record: RawDataItem) => {
        if (editingId === record.id) {
          return (
            <Input.TextArea
              value={editAnswer}
              onChange={(e) => setEditAnswer(e.target.value)}
              autoSize={{ minRows: 2, maxRows: 6 }}
              maxLength={MAX_ANSWER_LENGTH}
              aria-label="Answer text"
            />
          );
        }
        return <span>{text || <Tag color="warning">No answer</Tag>}</span>;
      },
      onCell: (record: RawDataItem) => ({
        onDoubleClick: () => {
          if (record.answer && editingId !== record.id) {
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
      render: (_: unknown, record: RawDataItem) => (
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
      // Server-side filter (only_unapproved): a client-side filter would only
      // look at the current page.
      title: 'Status',
      key: 'status',
      width: 80,
      filters: [{ text: 'Pending only', value: 'pending' }],
      filterMultiple: false,
      filteredValue: onlyUnapproved ? ['pending'] : null,
      render: (_: unknown, record: RawDataItem) => (
        record.admin_approved === 1 ?
          <Tag color="success">OK</Tag> :
          <Tag color="default">Pending</Tag>
      ),
    },
    {
      // The API has no duplicate filter, so this one filters the current page.
      title: 'Duplicates',
      key: 'duplicates',
      width: 120,
      filters: [
        { text: 'Duplicate', value: 'duplicate' },
        { text: 'Original', value: 'original' },
      ],
      filteredValue: rawDuplicateFilter,
      onFilter: (value, record: RawDataItem) => {
        if (value === 'duplicate') return record.is_duplicate === true;
        return !record.is_duplicate;
      },
      render: (_: unknown, record: RawDataItem) => {
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
      render: (text: string | null) => formatDateTime(text),
      sorter: (a: RawDataItem, b: RawDataItem) =>
        serverDateToMillis(a.created_at) - serverDateToMillis(b.created_at),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 180,
      fixed: 'right',
      render: (_: unknown, record: RawDataItem) => (
        <Space size="small">
          {editingId === record.id ? (
            <>
              <Button
                size="small"
                type="primary"
                onClick={handleSaveEdit}
                loading={updateAnswerMutation.isPending}
                disabled={!editAnswer.trim()}
              >
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
                aria-label={record.answer ? 'Edit answer' : 'Write answer'}
                title={record.answer ? 'Edit answer' : 'Write answer'}
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
                  aria-label="Approve to training data"
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
                  aria-label="Delete raw data"
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
      // Filters the current page only (no server-side duplicate filter).
      title: 'Duplicates',
      key: 'duplicates',
      width: 150,
      filters: [
        { text: 'Duplicate', value: 'duplicate' },
        { text: 'Original', value: 'original' },
      ],
      onFilter: (value, record: TrainingDataItem) => {
        if (value === 'duplicate') return record.is_answer_duplicate === true;
        // is_answer_duplicate may be null for rows never checked: those render as "Original".
        return !record.is_answer_duplicate;
      },
      render: (_: unknown, record: TrainingDataItem) => {
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
      render: (text: string | null) => formatDateTime(text),
    },
    {
      title: 'Actions',
      key: 'actions',
      width: 120,
      render: (_: unknown, record: TrainingDataItem) => (
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
          onClick={() => logout()}
        >
          Logout
        </Button>
      </div>

      {statsQuery.isError && <QueryErrorAlert title="Failed to load statistics" error={statsQuery.error} />}

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
              <>
                {rawDataQuery.isError && (
                  <QueryErrorAlert title="Failed to load raw data" error={rawDataQuery.error} />
                )}
                <Table
                  columns={rawDataColumns}
                  dataSource={rawDataResponse?.data || []}
                  rowKey="id"
                  loading={rawDataQuery.isLoading || (rawDataQuery.isFetching && rawDataQuery.isPlaceholderData)}
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
                  onChange={(_pagination, filters) => {
                    const nextOnlyUnapproved = filters.status?.[0] === 'pending';
                    if (nextOnlyUnapproved !== onlyUnapproved) {
                      setOnlyUnapproved(nextOnlyUnapproved);
                      setRawDataPage(1);
                    }
                    setRawDuplicateFilter(filters.duplicates ?? null);
                  }}
                  rowClassName={(record) => {
                    if (record.admin_approved === 1) return 'approved-row';
                    if (!record.answer) return 'unanswered-row';
                    return '';
                  }}
                />
              </>
            ),
          },
          {
            key: 'training',
            label: 'Training Data',
            children: (
              <>
                {trainingDataQuery.isError && (
                  <QueryErrorAlert title="Failed to load training data" error={trainingDataQuery.error} />
                )}
                <Table
                  columns={trainingDataColumns}
                  dataSource={trainingDataResponse?.data || []}
                  rowKey="id"
                  loading={
                    trainingDataQuery.isLoading ||
                    (trainingDataQuery.isFetching && trainingDataQuery.isPlaceholderData)
                  }
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
              </>
            ),
          },
          {
            key: 'docs',
            label: 'Documentation',
            children: (
              <LazyScreen fullHeight={false}>
                <DocumentViewer />
              </LazyScreen>
            ),
          },
        ]}
      />
    </div>
  );
}

export default Dashboard;
