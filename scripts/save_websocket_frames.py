#!/usr/bin/env python3
"""Save frames from WebSocket to verify the UI is working"""
import asyncio
import websockets
import json
import base64
from pathlib import Path

async def save_frames(uri="ws://localhost:8765/ws", output_dir="websocket_frames"):
    """Connect to WebSocket and save frames"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected successfully!")
            
            # Send reset command
            print("\nSending reset command...")
            await websocket.send(json.dumps({"action": "reset"}))
            
            # Receive and save initial frame
            print("Waiting for initial frame...")
            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(message)
            
            print(f"\n=== Initial Frame ===")
            print(f"Mode: {data.get('mode', 'N/A')}")
            print(f"Reward: {data.get('reward', 'N/A')}")
            print(f"Pred R: {data.get('pred_reward', 'N/A')}")
            print(f"Pred V: {data.get('pred_value', 'N/A')}")
            
            # Save environment image
            if 'env_img' in data:
                env_data = base64.b64decode(data['env_img'])
                env_path = output_path / "env_initial.jpg"
                env_path.write_bytes(env_data)
                print(f"✓ Saved environment image: {env_path}")
            
            # Save world model image
            if 'wm_img' in data:
                wm_data = base64.b64decode(data['wm_img'])
                wm_path = output_path / "wm_initial.jpg"
                wm_path.write_bytes(wm_data)
                print(f"✓ Saved world model image: {wm_path}")
            
            # Get a few more frames
            for i in range(3):
                await websocket.send(json.dumps({"action": "none"}))
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                
                print(f"\n=== Frame {i+1} ===")
                print(f"Mode: {data.get('mode', 'N/A')}")
                print(f"Reward: {data.get('reward', 'N/A'):.4f}")
                print(f"Pred R: {data.get('pred_reward', 'N/A'):.4f}")
                print(f"Pred V: {data.get('pred_value', 'N/A'):.2f}")
                
                # Save last frame images
                if i == 2:
                    if 'env_img' in data:
                        env_data = base64.b64decode(data['env_img'])
                        env_path = output_path / "env_frame3.jpg"
                        env_path.write_bytes(env_data)
                        print(f"✓ Saved environment image: {env_path}")
                    
                    if 'wm_img' in data:
                        wm_data = base64.b64decode(data['wm_img'])
                        wm_path = output_path / "wm_frame3.jpg"
                        wm_path.write_bytes(wm_data)
                        print(f"✓ Saved world model image: {wm_path}")
            
            print("\n✓ All frames saved successfully")
            return True
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    import sys
    uri = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765/ws"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "websocket_frames"
    
    success = asyncio.run(save_frames(uri, output_dir))
    sys.exit(0 if success else 1)
