#!/usr/bin/env python3
"""Test WebSocket connection to Dreamer 4 World Model Explorer"""
import asyncio
import websockets
import json
import sys

async def test_connection(uri="ws://localhost:8765/ws"):
    """Test WebSocket connection and receive frames"""
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected successfully!")
            
            # Send reset command first
            print("\nSending reset command...")
            await websocket.send(json.dumps({"action": "reset"}))
            
            # Receive a few frames
            for i in range(5):
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)
                    
                    print(f"\nFrame {i+1}:")
                    print(f"  Mode: {data.get('mode', 'N/A')}")
                    print(f"  Step: {data.get('step', 'N/A')}")
                    print(f"  Return: {data.get('return', 'N/A')}")
                    print(f"  Reward: {data.get('reward', 'N/A')}")
                    print(f"  Pred R: {data.get('pred_reward', 'N/A')}")
                    print(f"  Pred V: {data.get('pred_value', 'N/A')}")
                    print(f"  Has env_img: {'env_img' in data}")
                    print(f"  Has wm_img: {'wm_img' in data}")
                    
                    if 'env_img' in data:
                        print(f"  env_img length: {len(data['env_img'])} chars")
                    if 'wm_img' in data:
                        print(f"  wm_img length: {len(data['wm_img'])} chars")
                    
                    # Send a "none" action to get the next frame
                    if i < 4:
                        await websocket.send(json.dumps({"action": "none"}))
                        
                except asyncio.TimeoutError:
                    print(f"\nTimeout waiting for frame {i+1}")
                    break
                except json.JSONDecodeError as e:
                    print(f"\nError decoding JSON: {e}")
                    break
            
            print("\n✓ WebSocket test completed successfully")
            return True
            
    except ConnectionRefusedError:
        print("✗ Connection refused - is the server running?")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    uri = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765/ws"
    success = asyncio.run(test_connection(uri))
    sys.exit(0 if success else 1)
