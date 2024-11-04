export type LLMEngine = {
    chat: (message: string, updateBotMsg?: any) => Promise<void>;
    destroy?: () => void;
    greeting?: Message;
    init: any;
  };
  
  export type PostToWorker = {
    type: 'init' | 'chat';
    msg: string;
  };
  
  export type ListenFromWorker = {
    type: 'init' | 'chat';
    msg: string;
  };
  
  export type SendToWorkerMessageEventData = {
    curConversationIndex: number;
    msg: string;
    workerHistoryMsg?: WorkerHistoryMsg;
    ifNewConverstaion?: boolean;
  };
  
  export type ResFromWorkerMessageEventData = {
    type: 'initing' | 'chatting' | 'stats';
    action: 'append' | 'updateLast';
    msg: string;
    ifError?: boolean;
    ifFinish?: boolean;
  };
  