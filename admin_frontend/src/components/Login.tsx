import React, { useState } from 'react';
import { Form, Input, Button, Card, Alert, message } from 'antd';
import { LockOutlined, UserOutlined } from '@ant-design/icons';
import { useAuth } from '../contexts/AuthContext';
import { DEFAULT_ADMIN_USERNAME } from '../services/api';

interface LoginFormValues {
  username: string;
  password: string;
}

const Login: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { login, sessionExpired } = useAuth();

  const onFinish = async (values: LoginFormValues) => {
    setLoading(true);
    setError(null);
    try {
      const result = await login(values.username, values.password);
      if (result.ok) {
        message.success('Login successful!');
      } else {
        setError(result.error);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      background: '#f0f2f5'
    }}>
      <Card
        title="Admin Login"
        style={{ width: 400 }}
        styles={{ header: { textAlign: 'center' } }}
      >
        {sessionExpired && !error && (
          <Alert
            type="warning"
            showIcon
            message="Your session is no longer valid. Please log in again."
            style={{ marginBottom: 16 }}
          />
        )}
        {error && (
          <Alert type="error" showIcon message={error} style={{ marginBottom: 16 }} />
        )}
        <Form<LoginFormValues>
          name="login"
          onFinish={onFinish}
          layout="vertical"
          initialValues={{ username: DEFAULT_ADMIN_USERNAME }}
        >
          <Form.Item
            label="Username"
            name="username"
            rules={[{ required: true, whitespace: true, message: 'Please enter the username!' }]}
          >
            <Input
              prefix={<UserOutlined />}
              placeholder="Admin username"
              size="large"
              autoComplete="username"
            />
          </Form.Item>

          <Form.Item
            label="Password"
            name="password"
            rules={[{ required: true, message: 'Please enter the password!' }]}
          >
            <Input.Password
              prefix={<LockOutlined />}
              placeholder="Enter admin password"
              size="large"
              autoComplete="current-password"
            />
          </Form.Item>

          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              loading={loading}
              block
              size="large"
            >
              Login
            </Button>
          </Form.Item>
        </Form>
      </Card>
    </div>
  );
};

export default Login;
