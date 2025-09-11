import logging

from mcp.server import FastMCP
from mcp.types import TextContent

# 日志相关配置
logging.basicConfig(
    level=logging.INFO,
    encoding="utf-8",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("calculator_mcp_server")

# 初始化FastMCP服务器，并指定服务的名称为：calculator
mcp_server = FastMCP(
    name="calculator_mcp_server",  # 服务名
    instructions="计算器",  # 服务描述
)


# 定义加法工具函数
@mcp_server.tool()
async def add(x: float, y: float) -> list[TextContent]:
    """
    加法运算
    :param x: 第一个数字
    :param y: 第二个数字
    :return:
    """
    res = x + y
    logger.info(f"加法运算 :{x} + {y} = {res}")
    return [TextContent(type="text", text=str(res))]


# 定义减法工具函数
@mcp_server.tool()
async def subtract(x: float, y: float) -> list[TextContent]:
    """
    减法运算
    :param x: 第一个数字
    :param y: 第二个数字
    :return:
    """
    res = x - y
    logger.info(f"减法运算:{x} - {y} = {res}")
    return [TextContent(type="text", text=str(res))]


# 定义乘法运算工具函数
@mcp_server.tool()
async def multiply(x: float, y: float) -> list[TextContent]:
    """
    乘法运算
    :param x: 第一个数字
    :param y: 第二个数字
    :return:
    """
    res = x * y
    logger.info(f"乘法运算 :{x} * {y} = {res}")
    return [TextContent(type="text", text=str(res))]


# 定义除法运算工具函数
@mcp_server.tool()
async def divide(x: float, y: float) -> list[TextContent]:
    """
    除法运算
    :param x: 第一个数字
    :param y: 第二个数字
    :return:
    """
    res = x / y
    logger.info(f"除法运算:{x} / {y} = {res}")
    return [TextContent(type="text", text=str(res))]


if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    mcp_server.run(transport='stdio')