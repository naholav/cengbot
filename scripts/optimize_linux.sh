#!/bin/bash

# Linux Performance Optimization Script for CengBot
# Optimizes system settings for AI workloads and production deployment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SYSTEM_CONFIG_DIR="/etc/systemd/system"

# Logging
LOG_FILE="$PROJECT_DIR/logs/optimization.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

info() {
    log "INFO: $1"
    echo -e "${BLUE}ℹ️  $1${NC}"
}

success() {
    log "SUCCESS: $1"
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    log "WARNING: $1"
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    log "ERROR: $1"
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check if running as root for system optimizations
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root for system optimizations"
    fi
}

# System kernel parameter optimization
optimize_kernel_params() {
    info "Optimizing kernel parameters for AI workloads..."
    
    # Create sysctl configuration
    cat > /etc/sysctl.d/99-cengbot-optimization.conf << EOF
# CengBot AI System Optimization

# Memory management
vm.swappiness = 1
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
vm.overcommit_memory = 1

# Network optimization
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.core.rmem_default = 262144
net.core.wmem_default = 262144
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.ipv4.tcp_congestion_control = bbr
net.core.netdev_max_backlog = 5000

# File system optimization
fs.file-max = 65536
fs.nr_open = 1048576

# Security limits
kernel.pid_max = 4194304
EOF

    # Apply sysctl changes
    sysctl --system
    success "Kernel parameters optimized"
}

# System limits optimization
optimize_system_limits() {
    info "Optimizing system limits..."
    
    # Create limits configuration
    cat > /etc/security/limits.d/99-cengbot.conf << EOF
# CengBot system limits

# User limits
ceng soft nofile 65536
ceng hard nofile 65536
ceng soft nproc 32768
ceng hard nproc 32768

# System limits
* soft memlock unlimited
* hard memlock unlimited
EOF

    success "System limits optimized"
}

# GPU optimization
optimize_gpu() {
    info "Optimizing GPU settings..."
    
    # Check if NVIDIA GPU is available
    if command -v nvidia-smi &> /dev/null; then
        # Set GPU persistence mode
        nvidia-smi -pm 1
        
        # Set maximum performance mode
        nvidia-smi -ac 877,1455  # Adjust for your GPU
        
        # Create GPU monitoring script
        cat > /usr/local/bin/gpu-optimize.sh << EOF
#!/bin/bash
# GPU optimization script for CengBot

# Set persistence mode
nvidia-smi -pm 1

# Set performance mode
nvidia-smi -ac 877,1455

# Set power limit (adjust as needed)
nvidia-smi -pl 300

# Enable ECC if supported
nvidia-smi -e 1 || true
EOF
        chmod +x /usr/local/bin/gpu-optimize.sh
        
        # Create systemd service for GPU optimization
        cat > /etc/systemd/system/gpu-optimize.service << EOF
[Unit]
Description=GPU Optimization for CengBot
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/gpu-optimize.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

        systemctl daemon-reload
        systemctl enable gpu-optimize.service
        systemctl start gpu-optimize.service
        
        success "GPU optimization configured"
    else
        warning "NVIDIA GPU not detected, skipping GPU optimization"
    fi
}

# CPU optimization
optimize_cpu() {
    info "Optimizing CPU settings..."
    
    # Set CPU governor to performance
    echo "performance" > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
    
    # Create CPU optimization script
    cat > /usr/local/bin/cpu-optimize.sh << EOF
#!/bin/bash
# CPU optimization script for CengBot

# Set CPU governor to performance
echo "performance" > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true

# Disable CPU idle states for better performance
for i in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    echo 1 > "\$i" 2>/dev/null || true
done

# Set CPU affinity for better performance
echo "CPU optimization applied"
EOF
    chmod +x /usr/local/bin/cpu-optimize.sh
    
    # Create systemd service for CPU optimization
    cat > /etc/systemd/system/cpu-optimize.service << EOF
[Unit]
Description=CPU Optimization for CengBot
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/cpu-optimize.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable cpu-optimize.service
    systemctl start cpu-optimize.service
    
    success "CPU optimization configured"
}

# Memory optimization
optimize_memory() {
    info "Optimizing memory settings..."
    
    # Configure huge pages
    echo "vm.nr_hugepages = 1024" >> /etc/sysctl.d/99-cengbot-optimization.conf
    
    # Set transparent huge pages
    echo "never" > /sys/kernel/mm/transparent_hugepage/enabled
    echo "never" > /sys/kernel/mm/transparent_hugepage/defrag
    
    # Create memory optimization script
    cat > /usr/local/bin/memory-optimize.sh << EOF
#!/bin/bash
# Memory optimization script for CengBot

# Disable transparent huge pages
echo "never" > /sys/kernel/mm/transparent_hugepage/enabled
echo "never" > /sys/kernel/mm/transparent_hugepage/defrag

# Configure NUMA policy
echo 0 > /proc/sys/kernel/numa_balancing

# Memory compaction
echo 1 > /proc/sys/vm/compact_memory

echo "Memory optimization applied"
EOF
    chmod +x /usr/local/bin/memory-optimize.sh
    
    # Create systemd service for memory optimization
    cat > /etc/systemd/system/memory-optimize.service << EOF
[Unit]
Description=Memory Optimization for CengBot
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/memory-optimize.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable memory-optimize.service
    systemctl start memory-optimize.service
    
    success "Memory optimization configured"
}

# I/O optimization
optimize_io() {
    info "Optimizing I/O settings..."
    
    # Find the root device
    ROOT_DEVICE=$(df / | tail -1 | awk '{print $1}' | sed 's/[0-9]*$//')
    
    # Set I/O scheduler
    echo "mq-deadline" > /sys/block/"$(basename $ROOT_DEVICE)"/queue/scheduler 2>/dev/null || true
    
    # Create I/O optimization script
    cat > /usr/local/bin/io-optimize.sh << EOF
#!/bin/bash
# I/O optimization script for CengBot

# Set I/O scheduler for all block devices
for dev in /sys/block/*/queue/scheduler; do
    echo "mq-deadline" > "\$dev" 2>/dev/null || true
done

# Optimize readahead
for dev in /sys/block/*/queue/read_ahead_kb; do
    echo "512" > "\$dev" 2>/dev/null || true
done

# Optimize queue depth
for dev in /sys/block/*/queue/nr_requests; do
    echo "64" > "\$dev" 2>/dev/null || true
done

echo "I/O optimization applied"
EOF
    chmod +x /usr/local/bin/io-optimize.sh
    
    # Create systemd service for I/O optimization
    cat > /etc/systemd/system/io-optimize.service << EOF
[Unit]
Description=I/O Optimization for CengBot
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/io-optimize.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable io-optimize.service
    systemctl start io-optimize.service
    
    success "I/O optimization configured"
}

# Process scheduling optimization
optimize_scheduling() {
    info "Optimizing process scheduling..."
    
    # Create scheduling optimization script
    cat > /usr/local/bin/sched-optimize.sh << EOF
#!/bin/bash
# Process scheduling optimization for CengBot

# Set CFS bandwidth
echo "1000000" > /proc/sys/kernel/sched_cfs_bandwidth_slice_us

# Optimize scheduling domains
echo "0" > /proc/sys/kernel/sched_migration_cost_ns

# Set RT throttling
echo "950000" > /proc/sys/kernel/sched_rt_runtime_us

echo "Scheduling optimization applied"
EOF
    chmod +x /usr/local/bin/sched-optimize.sh
    
    # Create systemd service for scheduling optimization
    cat > /etc/systemd/system/sched-optimize.service << EOF
[Unit]
Description=Scheduling Optimization for CengBot
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/sched-optimize.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable sched-optimize.service
    systemctl start sched-optimize.service
    
    success "Process scheduling optimization configured"
}

# Service optimization
optimize_services() {
    info "Optimizing CengBot services..."
    
    # Create optimized service files
    cat > $SYSTEM_CONFIG_DIR/cengbot-worker.service << EOF
[Unit]
Description=CengBot AI Worker
After=network.target rabbitmq-server.service
Requires=rabbitmq-server.service

[Service]
Type=simple
User=ceng
Group=ceng
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONPATH=$PROJECT_DIR/src
ExecStart=/home/ceng/miniconda3/bin/python src/ai_model_worker.py
Restart=always
RestartSec=10
StandardOutput=append:$PROJECT_DIR/logs/worker.log
StandardError=append:$PROJECT_DIR/logs/worker.log

# Performance optimizations
Nice=-10
IOSchedulingClass=1
IOSchedulingPriority=4
CPUSchedulingPolicy=1
CPUSchedulingPriority=50

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

    cat > $SYSTEM_CONFIG_DIR/cengbot-bot.service << EOF
[Unit]
Description=CengBot Telegram Bot
After=network.target rabbitmq-server.service
Requires=rabbitmq-server.service

[Service]
Type=simple
User=ceng
Group=ceng
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONPATH=$PROJECT_DIR/src
ExecStart=/home/ceng/miniconda3/bin/python src/telegram_bot_rabbitmq.py
Restart=always
RestartSec=5
StandardOutput=append:$PROJECT_DIR/logs/bot.log
StandardError=append:$PROJECT_DIR/logs/bot.log

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

    cat > $SYSTEM_CONFIG_DIR/cengbot-admin.service << EOF
[Unit]
Description=CengBot Admin API
After=network.target

[Service]
Type=simple
User=ceng
Group=ceng
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONPATH=$PROJECT_DIR/src
ExecStart=/home/ceng/miniconda3/bin/python src/admin_rest_api.py
Restart=always
RestartSec=5
StandardOutput=append:$PROJECT_DIR/logs/admin.log
StandardError=append:$PROJECT_DIR/logs/admin.log

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

    cat > $SYSTEM_CONFIG_DIR/cengbot-monitor.service << EOF
[Unit]
Description=CengBot System Monitor
After=network.target

[Service]
Type=simple
User=ceng
Group=ceng
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONPATH=$PROJECT_DIR/src
ExecStart=/home/ceng/miniconda3/bin/python src/system_monitor.py
Restart=always
RestartSec=10
StandardOutput=append:$PROJECT_DIR/logs/monitor.log
StandardError=append:$PROJECT_DIR/logs/monitor.log

# Resource limits
LimitNOFILE=65536
LimitNPROC=1024

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    systemctl daemon-reload
    
    success "CengBot services optimized"
}

# Main optimization function
main() {
    info "Starting Linux optimization for CengBot..."
    
    # Check prerequisites
    check_root
    
    # Run optimizations
    optimize_kernel_params
    optimize_system_limits
    optimize_gpu
    optimize_cpu
    optimize_memory
    optimize_io
    optimize_scheduling
    optimize_services
    
    # Apply changes
    info "Applying system changes..."
    sysctl --system
    
    success "Linux optimization completed successfully!"
    success "System restart recommended for all changes to take effect"
    
    # Show summary
    info "Summary of optimizations:"
    info "- Kernel parameters optimized for AI workloads"
    info "- System limits increased for better performance"
    info "- GPU optimization configured (if available)"
    info "- CPU governor set to performance mode"
    info "- Memory settings optimized"
    info "- I/O scheduler optimized"
    info "- Process scheduling optimized"
    info "- CengBot services optimized with proper resource limits"
    
    warning "Please restart the system to apply all optimizations"
}

# Run main function
main "$@"