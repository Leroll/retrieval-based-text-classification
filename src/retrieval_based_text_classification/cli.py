"""Command line interface."""
import click
import sys
import signal
import os
import psutil
import json
import time
from pathlib import Path

from retrieval_based_text_classification.config import Config

# add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

def get_pid_file() -> Path:
    """Get the PID file path in project temp directory."""
    temp_dir = PROJECT_ROOT / "temp"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir / "retrieval_based_text_classification.json"
    
@click.group()
def cli():
    """Retrieval based Text Classification CLI."""
    pass

@cli.command()
@click.option('--config', '-c', 'config_path', 
              help='Path to configuration file')
def serve(config_path):
    """Start the FastAPI server."""
    # 如果指定了配置文件路径，设置环境变量
    if config_path:
        os.environ["CONFIG_PATH"] = config_path
        click.echo(click.style(f"📄 Using config file: {config_path}", fg="cyan"))
    
    cfg = Config()
    app_config = cfg.get_config().app
    pid_file = get_pid_file()
    
    # check if service is already running
    if pid_file.exists():
        try:
            with open(pid_file, "r") as f:
                existing_info = json.load(f)
            # 检查主进程是否存在
            if psutil.pid_exists(existing_info["master_pid"]):
                click.echo(click.style(f"❌ Service already running (Master PID: {existing_info['master_pid']})", fg="red"))
                return
            else:
                pid_file.unlink()
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            pid_file.unlink()
    
    # 保存服务信息
    service_info = {
        "master_pid": os.getpid(),
        "host": app_config.host,
        "port": app_config.port,
        "workers": app_config.workers,
        "start_time": time.time(),
        "config_path": cfg.config_path  # 记录使用的配置文件路径
    }
    with open(pid_file, "w") as f:
        json.dump(service_info, f)
    click.echo(click.style(f"📝 Service info saved to: {pid_file}", fg="cyan"))
    
    # start server
    try:
        from .app import main
        main()
    finally:
        if pid_file.exists():
            pid_file.unlink()
            click.echo(click.style("🧹 Cleaned up service info file", fg="cyan"))

@cli.command()
@click.option("--force", "-f", is_flag=True, help="Force shutdown using system signal")
def stop(force: bool):
    """Stop the running FastAPI server."""
    pid_file = get_pid_file()
    
    if not pid_file.exists():
        click.echo(click.style("❌ No service info found", fg="red"))
        return
    
    try:
        with open(pid_file, "r") as f:
            service_info = json.load(f)
        
        master_pid = service_info["master_pid"]
        
        if not psutil.pid_exists(master_pid):
            click.echo(click.style("❌ Master process not found", fg="red"))
            pid_file.unlink()
            return
        
        # 获取主进程和所有子进程
        try:
            master_process = psutil.Process(master_pid)
            all_processes = [master_process] + master_process.children(recursive=True)
            
            # 先尝试优雅关闭
            signal_type = signal.SIGTERM if force else signal.SIGINT
            signal_name = "SIGTERM" if force else "SIGINT"
            
            # 发送信号到所有进程
            for proc in all_processes:
                try:
                    if proc.is_running():
                        proc.send_signal(signal_type)
                        click.echo(click.style(f"📤 Sent {signal_name} to PID {proc.pid}", fg="cyan"))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # 等待进程终止
            click.echo(click.style("⏳ Waiting for processes to terminate...", fg="yellow"))
            wait_timeout = 10 if not force else 5
            
            terminated = []
            for proc in all_processes:
                try:
                    proc.wait(timeout=wait_timeout)
                    terminated.append(proc.pid)
                    click.echo(click.style(f"✅ Process {proc.pid} terminated", fg="green"))
                except psutil.TimeoutExpired:
                    click.echo(click.style(f"⚠️  Process {proc.pid} did not terminate within {wait_timeout}s", fg="yellow"))
                except psutil.NoSuchProcess:
                    terminated.append(proc.pid)
                    click.echo(click.style(f"✅ Process {proc.pid} already terminated", fg="green"))
            
            # 如果有进程没有终止，使用 SIGKILL 强制杀死
            remaining_processes = []
            for proc in all_processes:
                try:
                    if proc.is_running():
                        remaining_processes.append(proc)
                except psutil.NoSuchProcess:
                    continue
            
            if remaining_processes:
                click.echo(click.style(f"🔨 Force killing {len(remaining_processes)} remaining processes...", fg="red"))
                for proc in remaining_processes:
                    try:
                        proc.kill()  # SIGKILL
                        click.echo(click.style(f"💀 Killed PID {proc.pid}", fg="red"))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # 再次等待
                for proc in remaining_processes:
                    try:
                        proc.wait(timeout=3)
                    except (psutil.TimeoutExpired, psutil.NoSuchProcess):
                        continue
            
            # 最终检查
            still_running = []
            for proc in all_processes:
                try:
                    if proc.is_running():
                        still_running.append(proc.pid)
                except psutil.NoSuchProcess:
                    continue
            
            if still_running:
                click.echo(click.style(f"❌ Failed to stop processes: {still_running}", fg="red"))
            else:
                click.echo(click.style("✅ All processes terminated successfully", fg="green"))
            
        except psutil.NoSuchProcess:
            click.echo(click.style("❌ Process not found", fg="red"))
        
        # 清理文件
        if pid_file.exists():
            pid_file.unlink()
            click.echo(click.style("🧹 Cleaned up service info file", fg="cyan"))
            
    except Exception as e:
        click.echo(click.style(f"❌ Error stopping service: {e}", fg="red"))

@cli.command()
def status():
    """Check service status."""
    pid_file = get_pid_file()
    
    click.echo(click.style(f"📝 PID file location: {pid_file}", fg="cyan"))
    
    if not pid_file.exists():
        click.echo(click.style("❌ No service info found", fg="red"))
        return
    
    try:
        with open(pid_file, "r") as f:
            service_info = json.load(f)
        
        master_pid = service_info["master_pid"]
        host = service_info["host"]
        port = service_info["port"]
        workers = service_info.get("workers", 1)
        config_path = service_info.get("config_path")  # 获取配置文件路径
        
        if psutil.pid_exists(master_pid):
            # 检查所有相关进程
            try:
                master_process = psutil.Process(master_pid)
                all_processes = [master_process] + master_process.children(recursive=True)
                
                click.echo(click.style(f"✅ Service running on {host}:{port}", fg="green"))
                if config_path:
                    click.echo(click.style(f"📄 Config file: {config_path}", fg="magenta"))
                else:
                    click.echo(click.style(f"📄 Using default configuration", fg="magenta"))
                click.echo(click.style(f"📊 Master PID: {master_pid}, Workers: {workers}", fg="cyan"))
                click.echo(click.style(f"🔧 Active processes: {len(all_processes)}", fg="cyan"))
                
                # 显示进程详情
                for i, process in enumerate(all_processes, 1):
                    try:
                        click.echo(click.style(f"   Worker {i}: PID {process.pid}", fg="white"))
                    except psutil.NoSuchProcess:
                        click.echo(click.style(f"   Worker {i}: Process ended", fg="yellow"))
                        
            except psutil.NoSuchProcess:
                click.echo(click.style("❌ Master process not found", fg="red"))
                pid_file.unlink()
        else:
            click.echo(click.style("❌ Service not running (stale info file)", fg="red"))
            pid_file.unlink()
            
    except Exception as e:
        click.echo(click.style(f"❌ Error checking status: {e}", fg="red"))

@cli.command("clean")
@click.option("--confirm", "-y", is_flag=True, help="Skip confirmation prompt")
def clean(confirm: bool):
    """Clean log files in the logs/ directory."""
    log_dir = PROJECT_ROOT / "logs"
    
    # 检查logs目录是否存在
    if not log_dir.exists():
        click.echo(click.style(f"❌ Log directory does not exist: {log_dir}", fg="red"))
        return
    
    # 查找所有.log文件
    log_files = list(log_dir.glob("*.log*"))
    
    if not log_files:
        click.echo(click.style("✅ No log files found to delete", fg="green"))
        return
    
    # 显示要删除的文件
    click.echo(click.style(f"📁 Found {len(log_files)} log file(s) to delete:", fg="cyan"))
    for log_file in log_files:
        click.echo(f"  - {log_file.relative_to(PROJECT_ROOT)}")
    
    # 确认删除
    if not confirm:
        if not click.confirm(click.style("❓ Do you want to delete these files?", fg="yellow")):
            click.echo(click.style("❌ Operation cancelled", fg="red"))
            return
    
    # 删除文件
    deleted_count = 0
    for log_file in log_files:
        try:
            log_file.unlink()
            click.echo(click.style(f"🗑️  Deleted: {log_file.relative_to(PROJECT_ROOT)}", fg="green"))
            deleted_count += 1
        except Exception as e:
            click.echo(click.style(f"❌ Failed to delete {log_file.name}: {e}", fg="red"))
    
    click.echo(click.style(f"✅ Successfully deleted {deleted_count} log file(s)", fg="green"))

@cli.command()
def version():
    """Show version information."""
    from . import __version__
    from . import __build_date__
    from . import __author__ 
    click.echo(click.style(f"🪐 Retrieval based Text Classification: ", fg="cyan", bold=True) + 
               click.style(f"v{__version__}", fg="green", bold=True))
    click.echo(click.style(f"📅 Build Date: ", fg="cyan", bold=True) +
               click.style(f"{__build_date__}", fg="green", bold=True)),
    click.echo(click.style(f"👷 Author: ", fg="cyan", bold=True)+
               click.style(f"{__author__}", fg="green", bold=True))

if __name__ == "__main__":
    cli()