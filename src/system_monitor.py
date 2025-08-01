#!/usr/bin/env python3
"""
System monitoring and logging for CengBot
"""

import os
import time
import json
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import threading
import signal
import sys

# Database imports
from database_models import SessionLocal, RawData, TrainingData

@dataclass
class SystemMetrics:
    """System metrics data structure"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    gpu_available: bool
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_available_gb": self.memory_available_gb,
            "disk_usage_percent": self.disk_usage_percent,
            "disk_free_gb": self.disk_free_gb,
            "gpu_available": self.gpu_available,
            "gpu_memory_used": self.gpu_memory_used,
            "gpu_memory_total": self.gpu_memory_total
        }

@dataclass
class DatabaseMetrics:
    """Database metrics data structure"""
    timestamp: datetime
    raw_data_count: int
    training_data_count: int
    approved_count: int
    pending_count: int
    duplicate_count: int
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "raw_data_count": self.raw_data_count,
            "training_data_count": self.training_data_count,
            "approved_count": self.approved_count,
            "pending_count": self.pending_count,
            "duplicate_count": self.duplicate_count
        }

class SystemMonitor:
    """System monitoring and logging class"""
    
    def __init__(self, log_interval: int = 60, max_log_age_days: int = 30):
        self.log_interval = log_interval
        self.max_log_age_days = max_log_age_days
        self.running = False
        self.monitor_thread = None
        
        # Setup logging
        self.setup_logging()
        
        # Create monitoring directories
        self.monitoring_dir = Path("/home/ceng/cu_ceng_bot/logs/monitoring")
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.system_log_file = self.monitoring_dir / "system_metrics.jsonl"
        self.database_log_file = self.monitoring_dir / "database_metrics.jsonl"
        self.alerts_log_file = self.monitoring_dir / "alerts.jsonl"
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_percent": 90.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "disk_free_gb": 2.0,
            "gpu_memory_percent": 90.0
        }
        
        self.logger.info("System monitor initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("SystemMonitor")
        
        if not self.logger.handlers:
            # Create file handler
            log_file = "/home/ceng/cu_ceng_bot/logs/system_monitor.log"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.WARNING)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # GPU metrics
            gpu_available = False
            gpu_memory_used = None
            gpu_memory_total = None
            
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_available = True
                    # Get GPU memory info
                    torch.cuda.synchronize()
                    gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3  # GB
                    gpu_memory_total = torch.cuda.max_memory_allocated(0) / 1024**3  # GB
            except Exception:
                pass
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024**3),
                gpu_available=gpu_available,
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total
            )
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def collect_database_metrics(self) -> DatabaseMetrics:
        """Collect database metrics"""
        try:
            db = SessionLocal()
            
            raw_data_count = db.query(RawData).count()
            training_data_count = db.query(TrainingData).count()
            approved_count = db.query(RawData).filter(RawData.admin_approved == 1).count()
            pending_count = db.query(RawData).filter(RawData.admin_approved == 0).count()
            duplicate_count = db.query(RawData).filter(RawData.is_duplicate == True).count()
            
            db.close()
            
            return DatabaseMetrics(
                timestamp=datetime.now(),
                raw_data_count=raw_data_count,
                training_data_count=training_data_count,
                approved_count=approved_count,
                pending_count=pending_count,
                duplicate_count=duplicate_count
            )
        except Exception as e:
            self.logger.error(f"Error collecting database metrics: {e}")
            return None
    
    def check_alerts(self, system_metrics: SystemMetrics, database_metrics: DatabaseMetrics):
        """Check for alert conditions"""
        alerts = []
        
        # System alerts
        if system_metrics:
            if system_metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
                alerts.append({
                    "type": "cpu_high",
                    "message": f"High CPU usage: {system_metrics.cpu_percent:.1f}%",
                    "value": system_metrics.cpu_percent,
                    "threshold": self.alert_thresholds["cpu_percent"]
                })
            
            if system_metrics.memory_percent > self.alert_thresholds["memory_percent"]:
                alerts.append({
                    "type": "memory_high",
                    "message": f"High memory usage: {system_metrics.memory_percent:.1f}%",
                    "value": system_metrics.memory_percent,
                    "threshold": self.alert_thresholds["memory_percent"]
                })
            
            if system_metrics.disk_usage_percent > self.alert_thresholds["disk_usage_percent"]:
                alerts.append({
                    "type": "disk_high",
                    "message": f"High disk usage: {system_metrics.disk_usage_percent:.1f}%",
                    "value": system_metrics.disk_usage_percent,
                    "threshold": self.alert_thresholds["disk_usage_percent"]
                })
            
            if system_metrics.disk_free_gb < self.alert_thresholds["disk_free_gb"]:
                alerts.append({
                    "type": "disk_low",
                    "message": f"Low disk space: {system_metrics.disk_free_gb:.1f}GB",
                    "value": system_metrics.disk_free_gb,
                    "threshold": self.alert_thresholds["disk_free_gb"]
                })
        
        # Database alerts
        if database_metrics:
            if database_metrics.pending_count > 100:
                alerts.append({
                    "type": "pending_high",
                    "message": f"High pending questions: {database_metrics.pending_count}",
                    "value": database_metrics.pending_count,
                    "threshold": 100
                })
        
        # Log alerts
        for alert in alerts:
            alert_data = {
                "timestamp": datetime.now().isoformat(),
                "level": "warning",
                **alert
            }
            
            self.log_alert(alert_data)
            self.logger.warning(f"ALERT: {alert['message']}")
    
    def log_metrics(self, system_metrics: SystemMetrics, database_metrics: DatabaseMetrics):
        """Log metrics to files"""
        try:
            # Log system metrics
            if system_metrics:
                with open(self.system_log_file, 'a') as f:
                    f.write(json.dumps(system_metrics.to_dict()) + '\\n')
            
            # Log database metrics
            if database_metrics:
                with open(self.database_log_file, 'a') as f:
                    f.write(json.dumps(database_metrics.to_dict()) + '\\n')
                    
        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")
    
    def log_alert(self, alert_data: Dict):
        """Log alert to file"""
        try:
            with open(self.alerts_log_file, 'a') as f:
                f.write(json.dumps(alert_data) + '\\n')
        except Exception as e:
            self.logger.error(f"Error logging alert: {e}")
    
    def cleanup_old_logs(self):
        """Clean up old log files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_log_age_days)
            
            # Clean up monitoring logs
            for log_file in [self.system_log_file, self.database_log_file, self.alerts_log_file]:
                if log_file.exists():
                    # Read file and filter out old entries
                    new_lines = []
                    with open(log_file, 'r') as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                timestamp = datetime.fromisoformat(data['timestamp'])
                                if timestamp > cutoff_date:
                                    new_lines.append(line)
                            except:
                                continue
                    
                    # Write back filtered lines
                    with open(log_file, 'w') as f:
                        f.writelines(new_lines)
                        
            self.logger.info(f"Cleaned up logs older than {self.max_log_age_days} days")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up logs: {e}")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        system_metrics = self.collect_system_metrics()
        database_metrics = self.collect_database_metrics()
        
        return {
            "system": system_metrics.to_dict() if system_metrics else None,
            "database": database_metrics.to_dict() if database_metrics else None,
            "monitoring": {
                "running": self.running,
                "log_interval": self.log_interval,
                "max_log_age_days": self.max_log_age_days
            }
        }
    
    def monitor_loop(self):
        """Main monitoring loop"""
        self.logger.info("Starting monitoring loop")
        
        while self.running:
            try:
                # Collect metrics
                system_metrics = self.collect_system_metrics()
                database_metrics = self.collect_database_metrics()
                
                # Log metrics
                self.log_metrics(system_metrics, database_metrics)
                
                # Check for alerts
                self.check_alerts(system_metrics, database_metrics)
                
                # Periodic cleanup (once per day)
                current_time = datetime.now()
                if current_time.hour == 2 and current_time.minute < 2:  # 2 AM
                    self.cleanup_old_logs()
                
                # Wait for next interval
                time.sleep(self.log_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def start(self):
        """Start monitoring"""
        if self.running:
            self.logger.warning("Monitor is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("System monitor started")
    
    def stop(self):
        """Stop monitoring"""
        if not self.running:
            return
        
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("System monitor stopped")

# Global monitor instance
monitor = SystemMonitor()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\\nShutdown signal received, stopping monitor...")
    monitor.stop()
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Starting CengBot System Monitor...")
    print("Press Ctrl+C to stop")
    
    # Start monitoring
    monitor.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()