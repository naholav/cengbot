import React, { useMemo, useState } from 'react';
import { Card, List, Button, Modal, Spin, Tag, Input, Typography, Alert, Empty } from 'antd';
import { FileTextOutlined, CodeOutlined, FileOutlined } from '@ant-design/icons';
import { useDocumentContent, useDocuments } from '../hooks/useApiQueries';
import { getErrorMessage } from '../services/api';
import { DocumentInfo } from '../types/api';
import { formatSize } from '../utils/format';

const { Text } = Typography;
const { Search } = Input;

/**
 * The API returns at most this much of a file (the last part of a log, the
 * first part of anything else) and prepends/appends a truncation note.
 */
export const PARTIAL_VIEW_BYTES = 1024 * 1024;

const getIcon = (type: string) => {
  switch (type) {
    case 'documentation':
      return <FileTextOutlined />;
    case 'script':
      return <CodeOutlined />;
    default:
      return <FileOutlined />;
  }
};

const getTagColor = (type: string) => {
  switch (type) {
    case 'documentation':
      return 'blue';
    case 'script':
      return 'green';
    case 'log':
      return 'orange';
    default:
      return 'default';
  }
};

const DocumentViewer: React.FC = () => {
  const [selectedDoc, setSelectedDoc] = useState<DocumentInfo | null>(null);
  const [searchText, setSearchText] = useState('');

  const documentsQuery = useDocuments();
  const contentQuery = useDocumentContent(selectedDoc?.path ?? null);

  const groupedDocuments = useMemo(() => {
    const needle = searchText.toLowerCase();
    const documents = documentsQuery.data ?? [];
    return documents
      .filter(doc =>
        doc.name.toLowerCase().includes(needle) ||
        doc.path.toLowerCase().includes(needle)
      )
      .reduce((acc, doc) => {
        if (!acc[doc.type]) {
          acc[doc.type] = [];
        }
        acc[doc.type].push(doc);
        return acc;
      }, {} as Record<string, DocumentInfo[]>);
  }, [documentsQuery.data, searchText]);

  const groups = Object.entries(groupedDocuments);
  const closeModal = () => setSelectedDoc(null);

  const renderList = () => {
    if (documentsQuery.isLoading) {
      return <Spin size="large" />;
    }
    if (documentsQuery.isError) {
      return (
        <Alert
          type="error"
          showIcon
          message="Failed to load documents"
          description={getErrorMessage(documentsQuery.error)}
          action={<Button size="small" onClick={() => documentsQuery.refetch()}>Retry</Button>}
        />
      );
    }
    if (groups.length === 0) {
      return <Empty description="No documents found" />;
    }
    return groups.map(([type, docs]) => (
      <div key={type} style={{ marginBottom: 24 }}>
        <h3 style={{ textTransform: 'capitalize', marginBottom: 8 }}>
          {type}s ({docs.length})
        </h3>
        <List
          dataSource={docs}
          rowKey="path"
          renderItem={(doc) => {
            const partial = doc.size > PARTIAL_VIEW_BYTES;
            return (
              <List.Item
                actions={[
                  <Button key="view" type="link" onClick={() => setSelectedDoc(doc)}>
                    View
                  </Button>
                ]}
              >
                <List.Item.Meta
                  avatar={getIcon(doc.type)}
                  title={
                    <span>
                      {doc.name} <Tag color={getTagColor(doc.type)}>{doc.type}</Tag>
                    </span>
                  }
                  description={
                    <span>
                      <Text type="secondary">{doc.path}</Text>
                      <Text type="secondary" style={{ marginLeft: 16 }}>
                        {formatSize(doc.size)}
                      </Text>
                      {partial && (
                        <Text type="warning" style={{ marginLeft: 16 }}>
                          {doc.type === 'log' ? 'large file: only the last 1 MB is shown' : 'large file: only the first 1 MB is shown'}
                        </Text>
                      )}
                    </span>
                  }
                />
              </List.Item>
            );
          }}
        />
      </div>
    ));
  };

  const renderContent = () => {
    if (contentQuery.isLoading) {
      return <Spin size="large" />;
    }
    if (contentQuery.isError) {
      return (
        <Alert
          type="error"
          showIcon
          message="Failed to load document content"
          description={getErrorMessage(contentQuery.error)}
        />
      );
    }
    return (
      <div style={{ maxHeight: '70vh', overflow: 'auto' }}>
        {/* Rendered as text (never as HTML), so file content cannot inject markup. */}
        <pre style={{
          background: '#f5f5f5',
          padding: '16px',
          borderRadius: '4px',
          whiteSpace: 'pre-wrap',
          wordWrap: 'break-word'
        }}>
          {contentQuery.data ?? ''}
        </pre>
      </div>
    );
  };

  return (
    <>
      <Card title="Documentation & Logs">
        <Search
          placeholder="Search documents..."
          onChange={(e) => setSearchText(e.target.value)}
          style={{ marginBottom: 16 }}
        />
        {renderList()}
      </Card>

      <Modal
        title={selectedDoc?.name}
        open={selectedDoc !== null}
        onCancel={closeModal}
        footer={[
          <Button key="close" onClick={closeModal}>
            Close
          </Button>
        ]}
        width="80%"
        style={{ top: 20 }}
      >
        {renderContent()}
      </Modal>
    </>
  );
};

export default DocumentViewer;
