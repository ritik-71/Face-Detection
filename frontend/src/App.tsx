import React, { useState } from 'react';
import { ThemeProvider, createTheme, Box, Drawer, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Typography, Divider, AppBar, Toolbar } from '@mui/material';
import { LayoutDashboard, Video, UserCheck, UserPlus, MessageSquare, ShieldAlert } from 'lucide-react';

import Dashboard from './pages/Dashboard';
import LiveMonitor from './pages/LiveMonitor';
import AttendancePanel from './pages/AttendancePanel';
import UserEnrollment from './pages/UserEnrollment';
import AIAssistant from './pages/AIAssistant';

// Enterprise Cyber Dark Theme Palette
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00f2fe',
    },
    secondary: {
      main: '#4facfe',
    },
    background: {
      default: '#080710',
      paper: '#0f0e1f',
    },
    text: {
      primary: '#f8fafc',
      secondary: '#94a3b8',
    },
  },
  typography: {
    fontFamily: "'Outfit', sans-serif",
  },
});

const drawerWidth = 260;

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'monitor' | 'attendance' | 'enroll' | 'assistant'>('dashboard');

  const menuItems = [
    { id: 'dashboard', text: 'Dashboard', icon: <LayoutDashboard size={20} /> },
    { id: 'monitor', text: 'Live stream Grid', icon: <Video size={20} /> },
    { id: 'attendance', text: 'Attendance Logs', icon: <UserCheck size={20} /> },
    { id: 'enroll', text: 'Face Registration', icon: <UserPlus size={20} /> },
    { id: 'assistant', text: 'AI Assistant', icon: <MessageSquare size={20} /> },
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />;
      case 'monitor':
        return <LiveMonitor />;
      case 'attendance':
        return <AttendancePanel />;
      case 'enroll':
        return <UserEnrollment />;
      case 'assistant':
        return <AIAssistant />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
        {/* Navigation Sidebar */}
        <Drawer
          variant="permanent"
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: {
              width: drawerWidth,
              boxSizing: 'border-box',
              bgcolor: 'background.paper',
              borderRight: '1px solid rgba(0, 242, 254, 0.15)',
            },
          }}
        >
          <Box sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <Box sx={{ p: 1, borderRadius: 2, bg: 'primary.main', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid #00f2fe' }}>
              <ShieldAlert size={24} color="#00f2fe" />
            </Box>
            <Box>
              <Typography variant="h6" sx={{ fontWeight: 800, letterSpacing: '0.5px', color: '#00f2fe' }}>
                ANTIGRAVITY
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 600 }}>
                Face Analytics v2026
              </Typography>
            </Box>
          </Box>
          <Divider sx={{ borderColor: 'rgba(0, 242, 254, 0.1)' }} />
          
          <List sx={{ px: 2, py: 3 }}>
            {menuItems.map((item) => (
              <ListItem key={item.id} disablePadding sx={{ mb: 1 }}>
                <ListItemButton
                  onClick={() => setActiveTab(item.id as any)}
                  selected={activeTab === item.id}
                  sx={{
                    borderRadius: 3,
                    transition: 'all 0.2s',
                    py: 1.5,
                    border: activeTab === item.id ? '1px solid rgba(0, 242, 254, 0.2)' : '1px solid transparent',
                    background: activeTab === item.id ? 'linear-gradient(135deg, rgba(79,172,254,0.1) 0%, rgba(0,242,254,0.05) 100%) !important' : 'transparent',
                    '&:hover': {
                      background: 'rgba(0, 242, 254, 0.05)',
                      transform: 'translateX(4px)'
                    },
                    '&.Mui-selected .MuiListItemIcon-root': {
                      color: '#00f2fe',
                    },
                    '&.Mui-selected .MuiListItemText-primary': {
                      fontWeight: 700,
                      color: '#00f2fe',
                    }
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 40, color: activeTab === item.id ? '#00f2fe' : 'text.secondary' }}>
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText primary={item.text} primaryTypographyProps={{ fontSize: '14px', fontWeight: 500 }} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Drawer>

        {/* Main Content Workspace */}
        <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          <AppBar position="static" color="transparent" elevation={0} sx={{ borderBottom: '1px solid rgba(0, 242, 254, 0.1)' }}>
            <Toolbar sx={{ justifyContent: 'space-between', px: 4 }}>
              <Typography variant="h5" sx={{ fontWeight: 700, letterSpacing: '-0.5px' }}>
                {menuItems.find(item => item.id === activeTab)?.text}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <div className="pulse-indicator"></div>
                  <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 600 }}>
                    SYSTEMS OPERATIONAL
                  </Typography>
                </Box>
              </Box>
            </Toolbar>
          </AppBar>
          <Box sx={{ p: 4, flexGrow: 1, overflowY: 'auto' }}>
            {renderContent()}
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}
