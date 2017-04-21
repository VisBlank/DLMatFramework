/*
 * https://docs.unity3d.com/Manual/UsingDLL.html
 * https://unity3d.com/pt/learn/tutorials/topics/scripting/writing-plugins
 * 
 */
using System;
using System.Text;
using System.Threading;
using System.IO;
using System.Net.Sockets;
using System.Reflection;
using System.Net;

// Used for list
using System.Collections.Generic;

namespace TCPServerPlugin
{
	// Delegate(aka function pointer) to help logging on unity console
	public delegate void DebugLogDelegate(String message);
	// Delegate used on unity to update the control information (ie: Steering angle, speed, etc...) 
	//public delegate void UpdateControls(float m1, float m2, float m3, float m4);
	public delegate void UpdateControls(List<float> commands);


	public class TCPServerUnity
	{
		// Handle the server socket communication
		private TcpListener tcpListener = null;

		// Listen for socket connections
		private Thread listenThread;
		private Boolean isActive = false;
		private DebugLogDelegate mDebug;
		private UpdateControls mUpdateControls;
		private String textMessageToBeSent;
		private byte[] byteArrayToBeSent;
		private Boolean stopListen;
		private float m1, m2, m3, m4;

		// Lock to avoid the unity engine thread and this thread to read/write at the same time
		private Object thisLock;

		public TCPServerUnity(int port, DebugLogDelegate pLogDelegate, UpdateControls pUpMotors ,Object pLock)
		{
			// Set Delegate to log
			mDebug += pLogDelegate;

			// Set Delegate to update Motors values
			mUpdateControls += pUpMotors;

			stopListen = false;

			// Set the lock between threads (Defined on the unity script)
			thisLock = pLock;

			// Create TcpListerner object
			this.tcpListener = new TcpListener(IPAddress.Any, port);

			// Doing like this we're not going to block ...
			this.listenThread = new Thread(new ThreadStart(ListenForClients));

			// Start thread
			this.listenThread.Start();
		}

		public void setByteArray(byte[] incommingByteArray, int size)
		{
			byteArrayToBeSent = new byte[size];
			// Only one thread can acees this at a time (That's why we want to unlock as fast as possible)
			lock (thisLock)
			{
				/*
                    Array src,
                    int srcOffset,
                    Array dst,
                    int dstOffset,
                    int count
                 */
				Buffer.BlockCopy(incommingByteArray, 0, byteArrayToBeSent, 0, size);
			}
		}

		public void setMessage(String message)
		{
			// Only one thread can acees this at a time
			lock (thisLock)
			{
				textMessageToBeSent = message;
			}
		}

		public void Stop()
		{
			lock(thisLock) {
				stopListen = true;
				this.tcpListener.Stop();
			}
		}


		private void ListenForClients()
		{
			mDebug("Starting Thread to lister for clients ...");            
			// Start the server
			this.tcpListener.Start();
			isActive = true;
			while (true)
			{
				Boolean locStopListen;
				lock(thisLock) {
					locStopListen = stopListen;
				}
				if (locStopListen) break;

				if (tcpListener.Pending())
				{
					// The accept block until a new tcp client is connected
					TcpClient client = this.tcpListener.AcceptTcpClient();
					mDebug("Client arrived ");            

					// Now that the client has been conected we create a new thread to handle this communication
					// ParameterizedThreadStart represent a method with one parameter returning void that executes on a thread
					Thread clientThread = new Thread(new ParameterizedThreadStart(HandleClientComm));
					clientThread.Start(client); 
				}
			}
		}

		private void HandleClientComm(object client)
		{
			TcpClient tcpClient = (TcpClient)client;
			NetworkStream clientStream = tcpClient.GetStream();
			ASCIIEncoding encoder = new ASCIIEncoding();
			byte[] message = new byte[4096];
			int bytesRead;
			Boolean askedToCloseSocket = false;

			// Watchout to do not block everything
			while(true)
			{
				try
				{
					//blocks until a client sends a message
					bytesRead = clientStream.Read(message, 0, 4096);
				} catch
				{
					// Socket error occured
					break;
				}
				if (bytesRead == 0)
				{
					// The client has disconnected from the server
					break;
				}

				// Command arrived nicelly
				String recv_command = encoder.GetString(message, 0, bytesRead);
				recv_command = recv_command.Trim();
				if (recv_command.Contains("motor|"))
				{
					//mDebug("Motor command received....");  
					string [] resp = recv_command.Split('|');
					//mDebug("Size elements: " + resp.Length);  
					// Parse commands and add to a list
					List<float> listCommands = new List<float>();
					for (int idx = 1; idx <= resp.Length-1; idx++) {
						listCommands.Add (float.Parse(resp[idx]));
					}
					// Call some delegate with parsed commands (This delegate will be defined on the game side)
					//mDebug("Call delegate");  
					mUpdateControls(listCommands);

					// Send some bytes
					byte[] buffer = encoder.GetBytes("Ok\r\n");
					clientStream.Write(buffer, 0, buffer.Length);
					clientStream.Flush();
				} else {
					switch(recv_command)
					{ 
						case "telemetria":
					{
						// Get the current message to be sent ....                            
						String locCurrentMessage;
						lock (thisLock) 
						{
							locCurrentMessage = textMessageToBeSent;
						}

						// Send the size (4 byte) On the network endianess (Same as Java)
						byte[] msg_bytesSizeByteArray = BitConverter.GetBytes(IPAddress.HostToNetworkOrder(locCurrentMessage.Length));
						// Send image size
						clientStream.Write(msg_bytesSizeByteArray, 0, msg_bytesSizeByteArray.Length);
						clientStream.Flush();

						// Send some bytes
						byte[] buffer = encoder.GetBytes(locCurrentMessage);
						clientStream.Write(buffer, 0, buffer.Length);
						clientStream.Flush();
					}
						break;

						case "termina":
					{
						askedToCloseSocket = true;
					}
						break;

						case "imagem":
						byte[] locByteArray;
						lock (thisLock) 
						{
							locByteArray = new byte[byteArrayToBeSent.Length];
							Buffer.BlockCopy(byteArrayToBeSent, 0, locByteArray, 0, byteArrayToBeSent.Length);
						}

						// Send the size (4 byte) On the network endianess (Same as Java)
						int sizeByteArray = locByteArray.Length;
						byte[] bytesSizeByteArray = BitConverter.GetBytes(IPAddress.HostToNetworkOrder(sizeByteArray));

						//mDebug("Sending first " + bytesSizeByteArray.Length + " bytes with the size:" + locByteArray.Length + " hex:" + ByteArrayToString(bytesSizeByteArray)); 

						// Send image size
						clientStream.Write(bytesSizeByteArray, 0, bytesSizeByteArray.Length);
						clientStream.Flush();

						// Send the data (a lot of bytes)
						clientStream.Write(locByteArray, 0, locByteArray.Length);
						clientStream.Flush();
						break;
					}

					if (askedToCloseSocket) {
						tcpClient.Close();
						mDebug("Closing socket thread....");  
						break;
					};
				}
			}          
		}

		public bool isServerActive()
		{
			return isActive;
		}

		public string ByteArrayToString(byte[] ba)
		{
			string hex = BitConverter.ToString(ba);
			return hex.Replace("-"," ");
		}

	}
}


