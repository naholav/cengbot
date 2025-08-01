import React, { useState, useEffect } from 'react';
import { Card, List, Button, Modal, Spin, Tag, Input, Typography } from 'antd';
import { FileTextOutlined, CodeOutlined, FileOutlined } from '@ant-design/icons';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';

const { Text, Paragraph } = Typography;
const { Search } = Input;

interface Document {
  path: string;
  name: string;
  type: 'documentation' | 'log' | 'script';
  size: number;
}

const DocumentViewer: React.FC = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [docContent, setDocContent] = useState<string>('');
  const [modalVisible, setModalVisible] = useState(false);
  const [contentLoading, setContentLoading] = useState(false);
  const [searchText, setSearchText] = useState('');
  const { authHeader } = useAuth();

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await axios.get<Document[]>('http://localhost:8001/docs/list', {
        headers: { Authorization: authHeader }
      });
      setDocuments(response.data);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    } finally {
      setLoading(false);
    }
  };

  const viewDocument = async (doc: Document) => {
    setSelectedDoc(doc);
    setModalVisible(true);
    setContentLoading(true);

    try {
      const response = await axios.get(`http://localhost:8001/docs/read`, {
        params: { path: doc.path },
        headers: { Authorization: authHeader }
      });
      setDocContent(response.data);
    } catch (error) {
      console.error('Failed to read document:', error);
      setDocContent('Failed to load document content');
    } finally {
      setContentLoading(false);
    }
  };

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

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const filteredDocuments = documents.filter(doc => 
    doc.name.toLowerCase().includes(searchText.toLowerCase()) ||
    doc.path.toLowerCase().includes(searchText.toLowerCase())
  );

  const groupedDocuments = filteredDocuments.reduce((acc, doc) => {
    if (!acc[doc.type]) {
      acc[doc.type] = [];
    }
    acc[doc.type].push(doc);
    return acc;
  }, {} as Record<string, Document[]>);

  return (
    <>
      <Card title="Documentation & Logs">
        <Search
          placeholder="Search documents..."
          onChange={(e) => setSearchText(e.target.value)}
          style={{ marginBottom: 16 }}
        />
        
        {loading ? (
          <Spin size="large" />
        ) : (
          <>
            {Object.entries(groupedDocuments).map(([type, docs]) => (
              <div key={type} style={{ marginBottom: 24 }}>
                <h3 style={{ textTransform: 'capitalize', marginBottom: 8 }}>
                  {type}s ({docs.length})
                </h3>
                <List
                  dataSource={docs}
                  renderItem={(doc) => (
                    <List.Item
                      actions={[
                        <Button type="link" onClick={() => viewDocument(doc)}>
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
                          </span>
                        }
                      />
                    </List.Item>
                  )}
                />
              </div>
            ))}
          </>
        )}
      </Card>

      <Modal
        title={selectedDoc?.name}
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setModalVisible(false)}>
            Close
          </Button>
        ]}
        width="80%"
        style={{ top: 20 }}
      >
        {contentLoading ? (
          <Spin size="large" />
        ) : (
          <div style={{ maxHeight: '70vh', overflow: 'auto' }}>
            <pre style={{ 
              background: '#f5f5f5', 
              padding: '16px', 
              borderRadius: '4px',
              whiteSpace: 'pre-wrap',
              wordWrap: 'break-word'
            }}>
              {docContent}
            </pre>
          </div>
        )}
      </Modal>
    </>
  );
};

export default DocumentViewer;