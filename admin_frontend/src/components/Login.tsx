import React, { useState } from 'react';
import { Form, Input, Button, Card, message } from 'antd';
import { LockOutlined } from '@ant-design/icons';
import { useAuth } from '../contexts/AuthContext';

const Login: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();

  const onFinish = async (values: { password: string }) => {
    setLoading(true);
    const success = await login(values.password);
    setLoading(false);

    if (success) {
      message.success('Login successful!');
    } else {
      message.error('Invalid password');
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
        headStyle={{ textAlign: 'center' }}
      >
        <Form
          name="login"
          onFinish={onFinish}
          autoComplete="off"
          layout="vertical"
        >
          <Form.Item
            label="Password"
            name="password"
            rules={[{ required: true, message: 'Please enter the password!' }]}
          >
            <Input.Password 
              prefix={<LockOutlined />}
              placeholder="Enter admin password"
              size="large"
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